#ifndef __SYSARR_H__
#define __SYSARR_H__

// PE_S, X_SCALE, V_SIZE, and TYPE_SEL are external
#define     BLOCK_N     (PE_S * PE_S)
#define     BLOCK_M     (X_SCALE * V_SIZE)
#define     BLOCK_K     (PE_S * PE_S)

#define VECTOR_FLOAT1_ZERO          0.0f
#define VECTOR_FLOAT2_ZERO          (float2)(0.0f, 0.0f)
#define VECTOR_FLOAT4_ZERO          (float4)(0.0f, 0.0f, 0.0f, 0.0f)
#define VECTOR_FLOAT8_ZERO          (float8)(VECTOR_FLOAT4_ZERO,VECTOR_FLOAT4_ZERO)
#define VECTOR_FLOAT16_ZERO         (float16)(VECTOR_FLOAT8_ZERO,VECTOR_FLOAT8_ZERO)

#define VECTOR_INT4_ZERO            (int4)(0, 0, 0, 0)
#define VECTOR_INT8_ZERO            (int8)(VECTOR_INT4_ZERO, VECTOR_INT4_ZERO)

#define VECTOR_LONG4_ZERO           (long4)(0, 0, 0, 0)
#define VECTOR_LONG8_ZERO           (long8)(VECTOR_LONG4_ZERO, VECTOR_LONG4_ZERO)

// Shift register size
#define SHIFT_REG_SIZE              (PE_S * PE_S)

// Number of messages per block of A, B, or C
#define N_AB_BLOCK_MSGS             (PE_S * PE_S * X_SCALE)
#define N_C_BLOCK_MSGS              (PE_S * PE_S * PE_S)

////////////////////////////////////////////////////////////////////////
// Types
////////////////////////////////////////////////////////////////////////
#if TYPE_SEL==1

typedef long type;

#if V_SIZE==4

#define VECTOR_ZERO         VECTOR_LONG4_ZERO
typedef long4 vtype;

#elif V_SIZE==8

#define VECTOR_ZERO         VECTOR_LONG8_ZERO
typedef long8 vtype;

#else

#error Unsupported V_SIZE

#endif

#elif TYPE_SEL==2

typedef float type;

#if V_SIZE==1

#define VECTOR_ZERO         VECTOR_FLOAT1_ZERO
typedef float vtype;

#elif V_SIZE==2

typedef float2 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT2_ZERO
#define VECTOR_FMT          "v2hlf"

#elif V_SIZE==4

typedef float4 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT4_ZERO

#elif V_SIZE==8

typedef float8 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT8_ZERO
#define VECTOR_FMT          "v8hlf"

#elif V_SIZE==16
typedef float16 vtype;
#define VECTOR_ZERO         VECTOR_FLOAT16_ZERO
#else
#error Unsupported V_SIZE
#endif

#elif TYPE_SEL==3

typedef int type;

#if V_SIZE==4

#define VECTOR_ZERO         VECTOR_INT4_ZERO
typedef int4 vtype;

#elif V_SIZE==8

#define VECTOR_ZERO         VECTOR_INT8_ZERO
typedef int8 vtype;

#else

#error Unsupported V_SIZE

#endif

#else

#error Unsupported TYPE_SEL

#endif


#endif
