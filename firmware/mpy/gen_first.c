// Host-side generator of FIRST sets for MicroPython's grammar: for every rule, the set of token
// kinds that can start a match (all tokens if the rule can match nothing). py/parse.c with
// MICROPY_PARSE_FIRST_SETS uses them to skip alternatives that cannot match the current token
// instead of pushing the rule, trying it and backtracking.
//
// Built with the port's own config and include path (the Makefile), so the rule ids and the
// token numbering are exactly the firmware's. Prints first_sets.h to stdout.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "py/mpconfig.h"
#include "py/lexer.h"

#define RULE_ACT_ARG_MASK       (0x0f)
#define RULE_ACT_KIND_MASK      (0x30)
#define RULE_ACT_OR             (0x10)
#define RULE_ACT_AND            (0x20)
#define RULE_ACT_LIST           (0x30)
#define RULE_ACT_ALLOW_IDENT    (0x40)
#define RULE_ACT_ADD_BLANK      (0x80)
#define RULE_ARG_KIND_MASK      (0xf000)
#define RULE_ARG_ARG_MASK       (0x0fff)
#define RULE_ARG_TOK            (0x1000)
#define RULE_ARG_RULE           (0x2000)
#define RULE_ARG_OPT_RULE       (0x3000)

enum {
#define DEF_RULE(rule, comp, kind, ...) RULE_##rule,
#define DEF_RULE_NC(rule, kind, ...)
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
    RULE_const_object,
#define DEF_RULE(rule, comp, kind, ...)
#define DEF_RULE_NC(rule, kind, ...) RULE_##rule,
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
    RULE_maximum_number_of,
};

static const uint8_t rule_act_table[] = {
#define or(n)                   (RULE_ACT_OR | n)
#define and(n)                  (RULE_ACT_AND | n)
#define and_ident(n)            (RULE_ACT_AND | n | RULE_ACT_ALLOW_IDENT)
#define and_blank(n)            (RULE_ACT_AND | n | RULE_ACT_ADD_BLANK)
#define one_or_more             (RULE_ACT_LIST | 2)
#define list                    (RULE_ACT_LIST | 1)
#define list_with_end           (RULE_ACT_LIST | 3)
#define DEF_RULE(rule, comp, kind, ...) kind,
#define DEF_RULE_NC(rule, kind, ...)
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
    0,  // RULE_const_object
#define DEF_RULE(rule, comp, kind, ...)
#define DEF_RULE_NC(rule, kind, ...) kind,
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
#undef or
#undef and
#undef and_ident
#undef and_blank
#undef one_or_more
#undef list
#undef list_with_end
};

#define tok(t)                  (RULE_ARG_TOK | MP_TOKEN_##t)
#define rule(r)                 (RULE_ARG_RULE | RULE_##r)
#define opt_rule(r)             (RULE_ARG_OPT_RULE | RULE_##r)
#define DEF_RULE(rule, comp, kind, ...) static const uint16_t args_##rule[] = { __VA_ARGS__ };
#define DEF_RULE_NC(rule, kind, ...) static const uint16_t args_##rule[] = { __VA_ARGS__ };
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
#undef tok
#undef rule
#undef opt_rule

static const struct { const uint16_t *args; int n; } rule_args[] = {
#define DEF_RULE(rule, comp, kind, ...) { args_##rule, sizeof args_##rule / sizeof args_##rule[0] },
#define DEF_RULE_NC(rule, kind, ...)
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
    { 0, 0 },  // RULE_const_object
#define DEF_RULE(rule, comp, kind, ...)
#define DEF_RULE_NC(rule, kind, ...) { args_##rule, sizeof args_##rule / sizeof args_##rule[0] },
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
};

static const char *rule_names[] = {
#define DEF_RULE(rule, comp, kind, ...) #rule,
#define DEF_RULE_NC(rule, kind, ...)
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
    "const_object",
#define DEF_RULE(rule, comp, kind, ...)
#define DEF_RULE_NC(rule, kind, ...) #rule,
#include "py/grammar.h"
#undef DEF_RULE
#undef DEF_RULE_NC
};

#define NR RULE_maximum_number_of
#define NT MP_TOKEN_NUMBER_OF
#define BYTES 16

static const uint16_t *args_of[NR];
static int nargs_of[NR];
static int nullable[NR];
static uint8_t first[NR][BYTES];

static void setbit(uint8_t *s, int t) { s[t >> 3] |= 1 << (t & 7); }
static int or_into(uint8_t *d, const uint8_t *s) {
    int changed = 0;
    for (int i = 0; i < BYTES; i++) { uint8_t v = d[i] | s[i]; changed |= v != d[i]; d[i] = v; }
    return changed;
}

int main(void) {
    if (NT > BYTES * 8) { fprintf(stderr, "too many token kinds\n"); return 1; }
    for (int r = 0; r < NR; r++) {
        args_of[r] = rule_args[r].args;
        nargs_of[r] = rule_args[r].n;
    }
    // nullable: least fixpoint. tokens never; opt args yes; OR: any; AND: all; LIST: its item.
    for (int changed = 1; changed;) {
        changed = 0;
        for (int r = 0; r < NR; r++) {
            if (nullable[r] || r == RULE_const_object) continue;
            int kind = rule_act_table[r] & RULE_ACT_KIND_MASK, n = nargs_of[r], v;
            if (kind == RULE_ACT_OR) {
                v = 0;
                for (int i = 0; i < n; i++) {
                    uint16_t a = args_of[r][i], k = a & RULE_ARG_KIND_MASK;
                    if (k == RULE_ARG_OPT_RULE || (k == RULE_ARG_RULE && nullable[a & RULE_ARG_ARG_MASK])) v = 1;
                }
            } else if (kind == RULE_ACT_AND) {
                v = 1;
                for (int i = 0; i < n; i++) {
                    uint16_t a = args_of[r][i], k = a & RULE_ARG_KIND_MASK;
                    if (k == RULE_ARG_TOK || (k == RULE_ARG_RULE && !nullable[a & RULE_ARG_ARG_MASK])) v = 0;
                }
            } else {  // LIST: at least one item
                uint16_t a = args_of[r][0], k = a & RULE_ARG_KIND_MASK;
                v = k == RULE_ARG_OPT_RULE || (k == RULE_ARG_RULE && nullable[a & RULE_ARG_ARG_MASK]);
            }
            if (v) { nullable[r] = 1; changed = 1; }
        }
    }
    // FIRST: least fixpoint
    for (int changed = 1; changed;) {
        changed = 0;
        for (int r = 0; r < NR; r++) {
            if (r == RULE_const_object) continue;
            int kind = rule_act_table[r] & RULE_ACT_KIND_MASK, n = nargs_of[r];
            uint8_t s[BYTES];
            memset(s, 0, BYTES);
            int lim = kind == RULE_ACT_AND ? n : kind == RULE_ACT_OR ? n : 1;
            for (int i = 0; i < lim; i++) {
                uint16_t a = args_of[r][i], k = a & RULE_ARG_KIND_MASK, x = a & RULE_ARG_ARG_MASK;
                int null = 0;
                if (k == RULE_ARG_TOK) setbit(s, x);
                else { or_into(s, first[x]); null = k == RULE_ARG_OPT_RULE || nullable[x]; }
                if (kind == RULE_ACT_AND && !null) break;  // later args only after a nullable prefix
            }
            changed |= or_into(first[r], s);
        }
    }
    printf("// generated by gen_first.c: FIRST sets of the grammar rules, %d token kinds, %d rules\n", (int)NT, (int)NR);
    printf("#define FIRST_SET_BYTES %d\n", BYTES);
    printf("static const uint8_t rule_first_table[%d][%d] = {\n", (int)NR, BYTES);
    int pruned = 0;
    for (int r = 0; r < NR; r++) {
        uint8_t row[BYTES];
        if (nullable[r] || r == RULE_const_object) memset(row, 0xff, BYTES);  // never prune
        else { memcpy(row, first[r], BYTES); pruned++; }
        printf("    {");
        for (int i = 0; i < BYTES; i++) printf("0x%02x%s", row[i], i + 1 < BYTES ? "," : "");
        printf("},  // %s%s\n", rule_names[r], nullable[r] ? " (nullable)" : "");
    }
    printf("};\n");
    fprintf(stderr, "first_sets: %d rules, %d prunable, %d token kinds\n", (int)NR, pruned, (int)NT);
    return 0;
}
