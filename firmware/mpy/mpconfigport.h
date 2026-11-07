#include <stdint.h>
#if defined(RV_CFG_min)
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_NONE)
#elif defined(RV_CFG_basic)
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_BASIC_FEATURES)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_DOUBLE)
#elif defined(RV_CFG_extra)
#ifndef MICROPY_PY_UCTYPES
#define MICROPY_PY_UCTYPES       (0)
#endif
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_DOUBLE)
#endif
// --- P4 bridge additions ---
// Features a model reaches for that BASIC_FEATURES leaves out (each a few KB of ROM; the modules
// come from extmod/, see the Makefile). The seed for `random` is a hash of the conversation's
// tool-call text so far (main.c), so different conversations get different streams and the
// trace still proves every roll.
#define MICROPY_PY_FSTRINGS               (1)
#define MICROPY_MODULE_BUILTIN_INIT       (1)
#define MICROPY_PY_RANDOM                 (1)
#define MICROPY_PY_RANDOM_EXTRA_FUNCS     (1)
unsigned long rv_random_seed(void);
#define MICROPY_PY_RANDOM_SEED_INIT_FUNC  (rv_random_seed())
#define MICROPY_PY_RE                     (1)
#define MICROPY_PY_RE_SUB                 (1)
#define MICROPY_PY_JSON                   (0)  // json.loads needs io.StringIO (MICROPY_PY_IO, off)
#define MICROPY_PY_HEAPQ                  (1)
#define MICROPY_PY_HASHLIB                (1)
#define MICROPY_PY_HASHLIB_SHA256         (1)
#define MICROPY_PY_HASHLIB_SHA1           (0)
#define MICROPY_PY_HASHLIB_MD5            (0)
#define MICROPY_PY_MATH_FACTORIAL         (1)
#define MICROPY_PY_MATH_ISCLOSE           (1)
#define MICROPY_PY_BUILTINS_POW3          (1)
#define MICROPY_PY_BUILTINS_ROUND_INT     (1)
#define MICROPY_PY_BUILTINS_STR_PARTITION (1)
#define MICROPY_PY_BUILTINS_STR_SPLITLINES (1)
#define MICROPY_PY_BUILTINS_STR_CENTER    (1)
#define MICROPY_PY_COLLECTIONS_DEQUE      (1)
#define MICROPY_PY_SYS_MAXSIZE            (1)
#define MICROPY_PY_BUILTINS_BYTES_HEX     (1)
#define MICROPY_PY_BUILTINS_STR_SLICE_STEP (1)  // s[::-1] (firmware/patches/str-slice-step.patch)
// frozen modules (datetime) are only found by the full import machinery, through sys.path
#define MICROPY_ENABLE_EXTERNAL_IMPORT    (1)
#define MICROPY_PY_SYS_PATH               (1)
#define MICROPY_PY_SYS_PATH_DEFAULT       ".frozen"
#define MICROPY_PY_BUILTINS_HELP          (0)
#define MICROPY_PY_BUILTINS_INPUT         (0)

// Deep recursion raises RecursionError instead of overrunning the 6 KB stack into the heap.
#define MICROPY_STACK_CHECK               (1)
// A budget on VM jumps per tool call: main.c longjmps out of the interpreter when it runs out,
// so runaway code cannot keep the machine from parking (and user try/except cannot catch it).
extern unsigned long rv_vm_budget;
void rv_vm_budget_exceeded(void);
#define MICROPY_VM_HOOK_LOOP { if (--rv_vm_budget == 0) rv_vm_budget_exceeded(); }
#define MICROPY_VM_HOOK_RETURN { if (--rv_vm_budget == 0) rv_vm_budget_exceeded(); }

#ifdef RV_NOCOMP
#ifndef MICROPY_ENABLE_COMPILER
#define MICROPY_ENABLE_COMPILER           (0)
#endif
#else
#ifndef MICROPY_ENABLE_COMPILER
#define MICROPY_ENABLE_COMPILER           (1)
#endif
#endif
#ifndef MICROPY_ENABLE_GC
#define MICROPY_ENABLE_GC                 (1)
#endif
#ifndef MICROPY_HELPER_REPL
#define MICROPY_HELPER_REPL               (1)
#endif
#ifndef MICROPY_LONGINT_IMPL
#define MICROPY_LONGINT_IMPL              (MICROPY_LONGINT_IMPL_MPZ)
#endif
#ifndef MICROPY_ENABLE_EXTERNAL_IMPORT
#define MICROPY_ENABLE_EXTERNAL_IMPORT    (0)
#endif
#ifndef MICROPY_ALLOC_PATH_MAX
#define MICROPY_ALLOC_PATH_MAX            (256)
#endif
#ifndef MICROPY_ALLOC_PARSE_CHUNK_INIT
#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)
#endif
#ifndef MICROPY_PY_SYS_MODULES
#define MICROPY_PY_SYS_MODULES            (0)
#endif
#ifndef MICROPY_PY_SYS_EXIT
#define MICROPY_PY_SYS_EXIT               (0)
#endif
#ifndef MICROPY_PY_SYS_PATH
#define MICROPY_PY_SYS_PATH               (0)
#endif
#ifndef MICROPY_PY_SYS_ARGV
#define MICROPY_PY_SYS_ARGV               (0)
#endif
#ifndef MICROPY_PY_SYS_STDFILES
#define MICROPY_PY_SYS_STDFILES           (0)
#endif
#ifndef MICROPY_PY_OS
#define MICROPY_PY_OS                     (0)
#endif
#ifndef MICROPY_PY_TIME
#define MICROPY_PY_TIME                   (0)
#endif
#ifndef MICROPY_VFS
#define MICROPY_VFS                       (0)
#endif
#ifndef MICROPY_READER_VFS
#define MICROPY_READER_VFS                (0)
#endif
#ifndef MICROPY_PY_IO
#define MICROPY_PY_IO                     (0)
#endif
#ifndef MICROPY_PY_SELECT
#define MICROPY_PY_SELECT                 (0)
#endif
#ifndef MICROPY_PY_MACHINE
#define MICROPY_PY_MACHINE                (0)
#endif
#ifndef MICROPY_PY_RANDOM
#define MICROPY_PY_RANDOM                 (0)
#endif
#ifndef MICROPY_GCREGS_SETJMP
#define MICROPY_GCREGS_SETJMP             (0)
#endif
#define MP_SSIZE_MAX                      (0x7fffffff)
#ifndef MICROPY_PERSISTENT_CODE_LOAD
#define MICROPY_PERSISTENT_CODE_LOAD      (1)
#endif
typedef long mp_off_t;
#include <alloca.h>
#define MICROPY_HW_BOARD_NAME "chipformer"
#define MICROPY_HW_MCU_NAME "rv32im_zba"
#define MICROPY_HEAP_SIZE (HEAP_KB * 1024)
#define MP_STATE_PORT MP_STATE_VM

