// MicroPython on the chipformer RV32 machine as the P4 bridge: a Python tool for Qwen.
//
// Boot once (the boot lands in the firmware snapshot, not in any conversation), then park. Each
// GO (after Qwen's </tool_call>) reads the tool call's text from the capture buffer (its i-th
// token at TEXT + 16 i, NUL-padded, i = 1..80), takes the `code` parameter, runs it in __main__
// (globals persist across calls, like a REPL: a def in one call is usable in the next), and prints
//   \n<tool_response>\nOUTPUT</tool_response>\n<think>\n\n</think>\n   (+ the HALT echo \n)
// Bare expression statements print their value (REPL semantics). Exceptions print as tracebacks.
// A budget on VM jumps (RV_VM_BUDGET) ends runaway code with a longjmp out of the interpreter so
// the machine always gets back to its park; output is capped at OUT_MAX bytes per call.
#include <setjmp.h>
#include <stdint.h>
#include <string.h>
#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/cstack.h"
#include "py/mperrno.h"
#include "shared/runtime/gchelper.h"
#include "py/mphal.h"
#include "rt.h"

#ifndef RV_VM_BUDGET
#define RV_VM_BUDGET 3500  // VM jumps + returns per call: ~0.9M instructions, under the
                            // host lock's 1M-record limit per run (chat.py / bridge.Session)
#endif
#ifndef OUT_MAX
#define OUT_MAX 1536  // bytes of output per call that reach Qwen's context
#endif

#define TEXT ((const volatile unsigned char *)0x8000)
#define TOO_LONG "error: call too long (limit 80 tokens). Resend shorter: no comments, data as one " \
                 "string like 'a b c'.split(), or split across calls (variables persist).\n"
#define ROWS 80
#define ROW 16
enum { TOOL_RESPONSE = 256, TOOL_RESPONSE_END, THINK, THINK_END };

extern char __heap_start[], __heap_end[];

static void put(int c) { rv_ecall2(1, c, 0, 0); }
static void puts_(const char *s) { while (*s) put(*s++); }

// ---- the capture buffer -> source text ------------------------------------------------------

static char text[ROWS * ROW + 1];
static unsigned text_len;
static int text_full;  // every row used: the call may be cut off at the window

static unsigned long text_hash = 2166136261u;  // FNV-1a over every call's text: random's seed

unsigned long rv_random_seed(void) { return text_hash; }

static void gather(void) {
    unsigned n = 0;
    int i;
    for (i = 1; i <= ROWS; i++) {
        const volatile unsigned char *r = TEXT + ROW * i;
        if (!r[0]) break;
        for (int k = 0; k < ROW; k++) {
            unsigned char c = r[k];
            if (!c) break;
            text[n++] = (char)c;
            text_hash = (text_hash ^ c) * 16777619u;
        }
    }
    text[n] = 0;
    text_len = n;
    text_full = i > ROWS;
}

static const char *find(const char *hay, unsigned n, const char *needle) {
    unsigned m = strlen(needle);
    for (unsigned i = 0; i + m <= n; i++)
        if (memcmp(hay + i, needle, m) == 0) return hay + i;
    return 0;
}

// Qwen's XML-ish call: <function=NAME>\n<parameter=code>\nCODE\n</parameter>\n</function>
// Returns the code's start, sets *len; 0 if there is no code parameter (*len = -1 if the window
// closed before </parameter>).
static const char *xml_param(const char *tag, int *len) {
    const char *p = find(text, text_len, tag);
    if (!p) return 0;
    p += strlen(tag);
    if (*p == '\n') p++;
    const char *e = find(p, text + text_len - p, "</parameter>");
    if (!e) {
        *len = -1;
        return p;
    }
    if (e > p && e[-1] == '\n') e--;
    *len = (int)(e - p);
    return p;
}

// JSON-style fallback: {"name": "python", "arguments": {"code": "..."}}. Unescapes in place
// into `code`; returns its length, or -1 if there is no "code" string.
static char code_buf[ROWS * ROW + 1];
static int hex4(const char *s) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
              : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return -1;
        v = v * 16 + d;
    }
    return v;
}
static int json_str(const char *key) {
    const char *p = find(text, text_len, key);
    if (!p) return -1;
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '\n') p++;
    if (*p != '"') return -1;
    p++;
    int n = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\' && *p) {
            char e = *p++;
            if (e == 'n') c = '\n';
            else if (e == 't') c = '\t';
            else if (e == 'r') c = '\r';
            else if (e == 'u') {
                int v = hex4(p);
                if (v < 0) return -1;
                p += 4;
                if (v < 0x80) c = (char)v;
                else if (v < 0x800) { code_buf[n++] = (char)(0xC0 | v >> 6); c = (char)(0x80 | (v & 63)); }
                else { code_buf[n++] = (char)(0xE0 | v >> 12); code_buf[n++] = (char)(0x80 | (v >> 6 & 63)); c = (char)(0x80 | (v & 63)); }
            } else c = e;  // \" \\ \/
        }
        code_buf[n++] = c;
    }
    if (*p != '"') return -1;
    code_buf[n] = 0;
    return n;
}

// ---- running Python --------------------------------------------------------------------------

static unsigned out_n;
static char out_last;
static jmp_buf budget_jmp;
unsigned long rv_vm_budget;  // counted down by MICROPY_VM_HOOK_LOOP (mpconfigport.h)

void rv_vm_budget_exceeded(void) {
    longjmp(budget_jmp, 1);
}

static int run_src(const char *src, size_t len) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, len, 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t parse_tree = mp_parse(lex, MP_PARSE_FILE_INPUT);
        mp_obj_t module_fun = mp_compile(&parse_tree, source_name, true);  // REPL: exprs print
        mp_call_function_0(module_fun);
        nlr_pop();
        return 0;
    }
    mp_obj_print_exception(&mp_plat_print, (mp_obj_t)nlr.ret_val);
    return 1;
}

static void run_call(void) {
    gather();
    int len;
    const char *code = xml_param("<parameter=code>", &len);
    if (code && len < 0) {
        puts_(TOO_LONG);
        return;
    }
    if (!code) {
        len = json_str("\"code\"");
        if (len < 0) {
            puts_(text_full ? TOO_LONG : "error: no code parameter; call python with code=...\n");
            return;
        }
        code = code_buf;
    }
    // the optional `data` parameter: long plain-text input, as the global str `data`
    int dlen;
    const char *data = xml_param("<parameter=data>", &dlen);
    if (!data || dlen < 0) {
        data = 0;
        if (code != code_buf) {  // (the JSON buffer holds the code when JSON was used)
            dlen = json_str("\"data\"");
            if (dlen >= 0) data = code_buf;
        }
    }
    if (data) {
        nlr_buf_t nlr;
        if (nlr_push(&nlr) == 0) {
            mp_store_global(qstr_from_str("data"), mp_obj_new_str(data, (size_t)dlen));
            nlr_pop();
        }
    }
    rv_vm_budget = RV_VM_BUDGET;
    out_n = 0;
    nlr_buf_t *top = MP_STATE_THREAD(nlr_top);
    if (setjmp(budget_jmp) == 0) {
        run_src(code, (size_t)len);
    } else {
        MP_STATE_THREAD(nlr_top) = top;  // unwound past the VM's handlers
        rv_vm_budget = RV_VM_BUDGET;
        if (out_n && out_last != '\n') put('\n');
        const char *msg = "RuntimeError: step budget exceeded (the loop ran too long for one call)\n";
        mp_hal_stdout_tx_strn(msg, strlen(msg));
    }
    if (out_n == 0) puts_("(no output)\n");
    else if (out_last != '\n') put('\n');
}

int main(void) {
    mp_cstack_init_with_sp_here(STACK_BYTES);
    gc_init(__heap_start, __heap_end);
    mp_init();
    rv_vm_budget = RV_VM_BUDGET;
    for (;;) {
        rv_ecall2(93, 0, 0, 0);  // park: HALT; the next GO resumes here
        put('\n');
        put(TOOL_RESPONSE);
        put('\n');
        run_call();
        put(TOOL_RESPONSE_END);
        put('\n');
        put(THINK);
        put('\n');
        put('\n');
        put(THINK_END);
        put('\n');
    }
}

void gc_collect(void) {
    gc_collect_start();
    gc_helper_collect_regs_and_stack();
    gc_collect_end();
}

int mp_hal_stdin_rx_chr(void) { return 4; }

mp_uint_t mp_hal_stdout_tx_strn(const char *str, mp_uint_t len) {
    for (mp_uint_t i = 0; i < len; i++) {
        if (out_n == OUT_MAX) {
            puts_("\n...[output truncated]\n");
            out_n++;
        }
        if (out_n > OUT_MAX) continue;
        if (str[i] == '\r') continue;  // print() writes \r\n; Qwen reads bytes as tokens
        rv_putchar(str[i]);
        out_last = str[i];
        out_n++;
    }
    return len;
}

mp_lexer_t *mp_lexer_new_from_file(qstr filename) { mp_raise_OSError(MP_ENOENT); }
mp_import_stat_t mp_import_stat(const char *path) { return MP_IMPORT_STAT_NO_EXIST; }
void nlr_jump_fail(void *val) { for (;;) rv_ecall2(93, 3, 0, 0); }
void MP_NORETURN __fatal_error(const char *msg) { for (;;) rv_ecall2(93, 4, 0, 0); }
