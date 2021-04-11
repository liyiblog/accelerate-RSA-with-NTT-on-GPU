#include "common.c"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
typedef long long ll;

const int  P = 23068673, G = 3;
int ebits, dbits, nbits;

//input:a k 
//output:a^k mod P
inline ll power(ll a, ll k) {
	ll base = 1;
	for (; k; k >>= 1) {
		if (k & 1) base = (base * a) % P;
		a = (a * a) % P;
	}
	return base;
}

//input:bignum bn 
//output:bits of bn
int bignum_numbits(uint32_t* bn) {

	register int i = (64 << 5) - 1;
	for (; i > 0; --i)
	{
		if ((bn[i >> 5] >> (i & 0b11111)) & 1)
			return i + 1;
	}
	return 0;
}

//read bignum from string
void bignum_from_string(uint32_t* bn, char* str, int nhex)
{
	memset(bn, 0, nhex / 2);

	uint32_t tmp;                        
	int i = nhex - 8;				/* index into string */
	int j = 0;						/* index into array */

	/* reading last hex-byte "MSB" from string first -> big endian */
	/* MSB ~= most significant byte / block ? :) */
	while (i >= 0)
	{
		tmp = 0;
		sscanf(&str[i], "%8x", &tmp);//read one word
		bn[j] = tmp;
		i -= 8; /* step WORD_SIZE hex-byte(s) back in the string. */
		j += 1; /* step one element forward in the array. */
	}
}

__global__ void PrintOnGpu(uint32_t* X)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("[%d]:%u ", tid, X[tid]);
}

//input:a b 
//output:a + b 
__global__ void AddOnGPU(uint32_t* Device_A, uint32_t* Device_B)
{
	__shared__ int flag;
	flag = 0;
	extern __shared__ int carry[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t temp;
	carry[tid] = 0;
	temp = Device_A[tid] + Device_B[tid];
	carry[tid + 1] = temp >> 32;
	if (carry[tid])
		flag = 1;
	Device_A[tid] = temp;
	for (; flag == 1; )
	{
		flag = 0;
		temp = Device_A[tid] + carry[tid];
		carry[tid + 1] = temp >> 32;
		if (carry[tid])
			flag = 1;
		Device_A[tid] = temp;
	}
}

//input:a b 
//output:a - b 
__global__ void SubOnGPU(uint32_t* Device_A, uint32_t* Device_B, int* cmp)
{
	if (*cmp != -1)
	{		
		extern __shared__ uint32_t borrow[];
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		uint64_t res;
		borrow[tid] = 0;

		res = (uint64_t)Device_A[tid] + 0x100000000 - (uint64_t)Device_B[tid];
		borrow[tid + 1] = (res >> 32) ^ 1;
		Device_A[tid] = res;
		__syncthreads();
		for (; borrow[tid] == 1; )
		{		
			res = (uint64_t)Device_A[tid] + 0x100000000 - (uint64_t)borrow[tid];
			borrow[tid] = 0;
			borrow[tid + 1] = (res >> 32) ^ 1;
			Device_A[tid] = res;
		}
		__syncthreads();
	}
}

//input:a b flag 
//output: flag = (a > b) ? 1:0:-1
__global__ void CmpOnGPU(uint32_t* Device_A, uint32_t* Device_B,int size,int* flag)
{
	*flag = 0;
	for (int i = size; i >= 0; i--)
	{
		if (Device_A[i] > Device_B[i])
		{
			*flag = 1;
			break;
		}
		else if (Device_A[i] < Device_B[i])
		{
			*flag = -1;
			break;
		}
	}
}

//ntt
__global__ void NttOnGPU(uint32_t* Device_A, ll* g, int limit) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t x, y;
	int j, m;
	for (m = 1; m < limit && m < 32; m <<= 1) {
		j = tid / m % 2;
		x = (ll)Device_A[tid] * g[j * limit / (2 * m) * (tid % m)] % P;
		Device_A[tid] = ((1 - 2 * j) * x + __shfl_xor(x, m) + j * P) % P;
	}
	for (; m < limit; m <<= 1) {
		j = tid / m % 2;
		if (j)
		{
			x = (ll)Device_A[tid] * g[limit / (2 * m) * (tid % m)] % P;
			Device_A[tid] = x;
		}
		else
		{
			x = Device_A[tid];
		}
		__syncthreads();
		y = Device_A[tid ^ m];
		Device_A[tid] = ((1 - 2 * j) * x + y + j * P) % P;
	}
}

//inverse ntt
__global__ void _NttOnGPU(uint32_t* Device_A, ll* g, int limit) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t x, y;
	int j, m;
	for (m = 1; m < limit && m < 32; m <<= 1) {
		j = tid / m % 2;
		x = (ll)Device_A[tid] * g[j * ((limit - (limit / (2 * m) * (tid % m))) % limit)] % P;
		Device_A[tid] = ((1 - 2 * j) * x + __shfl_xor(x, m) + j * P) % P;
	}

	for (; m < limit; m <<= 1) {
		j = tid / m % 2;
		if (j)
		{
			x = (ll)Device_A[tid] * g[(limit - (limit / (2 * m) * (tid % m))) % limit] % P;
			Device_A[tid] = x;
		}
		else
		{
			x = Device_A[tid];
		}
		__syncthreads();
		y = Device_A[tid ^ m];
		Device_A[tid] = ((1 - 2 * j) * x + y + j * P) % P;
	}
}

//input:a b 
//output:a * b mod P
__global__ void MulOnGPU(uint32_t* Device_A, uint32_t* Device_B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Device_A[tid] = ((ll)Device_A[tid] * (ll)Device_B[tid]) % P;
}

//input:ntt(a) ntt(b) 
//output:ntt(a) + ntt(b)
__global__ void NttAddOnGPU(uint32_t* Device_A, uint32_t* Device_B) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Device_A[tid] = Device_A[tid] + Device_B[tid];
}

//butterfly in gpu
__global__ void ButOnGPU(uint32_t* Device_A, int* D_rev) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Device_A[tid] = Device_A[D_rev[tid]];
}

//input:a inv output:a/n mod p
__global__ void MulDivOnGPU(uint32_t* Device_A, ll inv) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Device_A[tid] = (ll)Device_A[tid] * inv % P;
}

//bignum to polynomial
__global__ void BignumToPolynomial(uint8_t* x, uint32_t* X, int limit) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	X[tid] = x[tid];
	X[tid + limit] = 0;
}

//polynomial to bignum and mod R
__global__ void PolynomialToBignum(uint8_t* x, uint32_t* X, int offset, int nbytes, int nbits) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (nbits != 0)
	{
		int z = 8 - nbits;
		x[tid] = ((uint8_t)X[tid + nbytes]) >> nbits;
		x[tid] += ((uint8_t)X[tid + nbytes + 1]) << z;
	}
	else
	{
		x[tid] = X[tid + nbytes];
	}
	x[tid + offset] = 0;

}

//polynomial mod R
__global__ void PolynominalModR(uint32_t* X, uint32_t* Out, int limit,int nbits) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	Out[tid] = X[tid];
	Out[tid + limit] = 0;
	Out[nbits >> 3] = Out[nbits >> 3] % (1 << (nbits & 0b111));
	//Out[limit - 1] = Out[limit - 1] & 0b1111;
}

//calculate carry 
__global__ void PolynominalCarry(uint32_t* X, int limit)
{
	for (int i = 0; i < limit; i++)
	{
		X[i + 1] += X[i] >> 8;
		X[i] = X[i] & 0xFF;
	}
}

//global val

uint32_t* nn, * e, * d, * r2m, * _n, * val;
uint8_t* x, * y;
uint32_t* n_ntt, * _n_ntt, * M;
ll inv;
int* d_rev;
ll* d_root;
ll root[1 << 11] = { 0 };
int rev[1 << 12] = { 0 };

cudaStream_t stream[16];
uint32_t* XX[16], * YY[16];
uint8_t* xx[16], * yy[16];
uint32_t* Out[16];
int* flag[16];

// montgomery mul with ntt
void NTTMonMulOnGPU(cudaStream_t stream, uint8_t* x, uint8_t* y, uint32_t* out, uint32_t* X, uint32_t* Y, int limit, int* flag)
{
	BignumToPolynomial << <1, limit / 2, 0, stream >> > (x, X, limit / 2);
	BignumToPolynomial << <1, limit / 2, 0, stream >> > (y, Y, limit / 2);

	/*
	if (test)
	{
		PrintOnGpu << <1, limit, 0, stream >> > (X);
		cudaStreamSynchronize(stream);
		printf("\n\n");
		PrintOnGpu << <1, limit, 0, stream >> > (Y);
		cudaStreamSynchronize(stream);
		printf("\n\n");
	}
	*/

	// X = X * Y
	ButOnGPU << <1, limit, 0, stream >> > (X, d_rev);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (X, d_root, limit);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (X, Y);
	ButOnGPU << <1, limit, 0, stream >> > (X, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (X, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (X, inv);
	PolynominalCarry << <1, 1, 0, stream >> > (X, limit / 2);

	// Y = X * _n mod R
	PolynominalModR << <1, limit / 2, 0, stream >> > (X, Y, limit / 2, nbits);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, _n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);
	PolynominalCarry << <1, 1, 0, stream >> > (Y, limit / 2);
	PolynominalModR << <1, limit / 2, 0, stream >> > (Y, Y, limit / 2, nbits);

	// Y = Y * n
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);

	// out = (X + Y) / R 
	NttAddOnGPU << <1, limit, 0, stream >> > (X, Y);
	PolynominalCarry << <1, 1, 0, stream >> > (X, limit - 1);
	PolynomialToBignum << <1, limit / 2, 0, stream >> > ((uint8_t*)out, X, limit / 2, nbits >> 3, nbits & 0b111);

	// if out >= n ,return out - n,else return out
	CmpOnGPU << <1, 1, 0, stream >> > (out, M, (limit - 1) / 8, flag);
	SubOnGPU << < 1, (limit + 7) / 8, ((limit + 7) / 8 + 1) * sizeof(uint32_t), stream >> > (out, M, flag);
}

// montgomery red with ntt
void NTTMonRedOnGPU(cudaStream_t stream, uint8_t* y, uint32_t* out, uint32_t* X, uint32_t* Y, int limit, int* flag)
{
	// X = Y * 1
	BignumToPolynomial << <1, limit / 2, 0, stream >> > (y, X, limit / 2);
	BignumToPolynomial << <1, limit / 2, 0, stream >> > (y, Y, limit / 2);

	// Y = Y * _n mod R
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, _n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);
	PolynominalCarry << <1, 1, 0, stream >> > (Y, limit / 2);
	PolynominalModR << <1, limit / 2, 0, stream >> > (Y, Y, limit / 2, nbits);

	// Y = Y * n
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);

	// out = (X + Y) / R
	NttAddOnGPU << <1, limit, 0, stream >> > (X, Y);
	PolynominalCarry << <1, 1, 0, stream >> > (X, limit - 1);
	PolynomialToBignum << <1, limit / 2, 0, stream >> > ((uint8_t*)out, X, limit / 2, nbits >> 3, nbits & 0b111);

	// if out >= n ,return out - n,else return out
	CmpOnGPU << <1, 1, 0, stream >> > (out, M, (limit - 1) / 8, flag);
	SubOnGPU << < 1, (limit + 7) / 8, ((limit + 7) / 8 + 1) * sizeof(uint32_t), stream >> > (out, M, flag);
}

// montgomery squ with ntt
void NTTMonSquOnGPU(cudaStream_t stream, uint8_t* x, uint32_t* out, uint32_t* X, uint32_t* Y, int limit, int* flag)
{
	BignumToPolynomial << <1, limit / 2, 0, stream >> > (x, X, limit / 2);

	// X = X * Y
	ButOnGPU << <1, limit, 0, stream >> > (X, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (X, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (X, X);
	ButOnGPU << <1, limit, 0, stream >> > (X, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (X, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (X, inv);;
	PolynominalCarry << <1, 1, 0, stream >> > (X, limit / 2);

	// Y = X * _n mod R
	PolynominalModR << <1, limit / 2, 0, stream >> > (X, Y, limit / 2, nbits);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, _n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);
	PolynominalCarry << <1, 1, 0, stream >> > (Y, limit / 2);
	PolynominalModR << <1, limit / 2, 0, stream >> > (Y, Y, limit / 2, nbits);

	// Y = Y * n
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulOnGPU << <1, limit, 0, stream >> > (Y, n_ntt);
	ButOnGPU << <1, limit, 0, stream >> > (Y, d_rev);
	_NttOnGPU << <1, limit, 0, stream >> > (Y, d_root, limit);
	MulDivOnGPU << <1, limit, 0, stream >> > (Y, inv);

	// out = (X + Y) / R 
	NttAddOnGPU << <1, limit, 0, stream >> > (X, Y);
	PolynominalCarry << <1, 1, 0, stream >> > (X, limit - 1);
	PolynomialToBignum << <1, limit / 2, 0, stream >> > ((uint8_t*)out, X, limit / 2, nbits >> 3, nbits & 0b111);

	// if out >= n ,return out - n,else return out
	CmpOnGPU << <1, 1, 0, stream >> > (out, M, (limit - 1) / 8, flag);
	SubOnGPU << < 1, (limit + 7) / 8, ((limit + 7) / 8 + 1) * sizeof(uint32_t), stream >> > (out, M, flag);
}


//input x e limit:nttitems ebits s:stream idex
//output:x^e mod n
void MonExpOnGPU(uint8_t* x, uint32_t* e, int limit, int ebits, int s)
{
	// x=x*r mod n
	NTTMonMulOnGPU(stream[s], x, y, (uint32_t*)xx[s], XX[s], YY[s], limit, flag[s]);

	// y=r mod n
	NTTMonRedOnGPU(stream[s], y, (uint32_t*)yy[s], XX[s], YY[s], limit, flag[s]);

	for (int i = 0; i < ebits - 1; i++)
	{
		/*
		if (i < 0)
		{
			test = 0;
			PrintOnGpu << <1, limit / 4, 0, stream[s] >> > ((uint32_t*)xx[s]);
			cudaStreamSynchronize(stream[s]);
			printf("\n\n");
		}
		*/
		if ((e[i >> 5] >> (i & 0b11111)) & 1)
			NTTMonMulOnGPU(stream[s], xx[s], yy[s], (uint32_t*)yy[s], XX[s], YY[s], limit, flag[s]);
		//NTTMonMulOnGPU(stream[s], xx[s], xx[s], (uint32_t*)xx[s], XX[s], YY[s], limit, flag[s], test);
		NTTMonSquOnGPU(stream[s], xx[s], (uint32_t*)xx[s], XX[s], YY[s], limit, flag[s]);
		
	}

	NTTMonMulOnGPU(stream[s], xx[s], yy[s], (uint32_t*)yy[s], XX[s], YY[s], limit, flag[s]);
	NTTMonRedOnGPU(stream[s], yy[s], Out[s], XX[s], YY[s], limit, flag[s]);
	
}

//Pre calculation bignum to ntt
void NumToNtt(uint8_t* x,uint32_t* X, int* d_rev, ll* d_root, int limit)
{
	BignumToPolynomial << <1, limit / 2 >> > (x, X, limit / 2);
	ButOnGPU << <1, limit>> > (X, d_rev);
	NttOnGPU << <1, limit>> > (X, d_root, limit);
}

int initongpu()
{
	CHECK(cudaMallocHost((uint32_t * *)& nn, 256));
	CHECK(cudaMallocHost((uint32_t * *)& e, 256));
	CHECK(cudaMallocHost((uint32_t * *)& d, 256));
	CHECK(cudaMallocHost((uint32_t * *)& r2m, 256));
	CHECK(cudaMallocHost((uint32_t * *)& _n, 256));
	CHECK(cudaMallocHost((uint32_t * *)& val, 256));
	
	
	CHECK(cudaMalloc((uint32_t * *)& n_ntt, 2048));
	CHECK(cudaMalloc((uint32_t * *)& _n_ntt, 2048));
	CHECK(cudaMalloc((uint8_t * *)& x, 256));
	CHECK(cudaMalloc((uint8_t * *)& y, 256));
	CHECK(cudaMalloc((uint32_t * *)& M, 256));
	CHECK(cudaMemset(x, 0, 256));
	CHECK(cudaMemset(y, 0, 256));
	CHECK(cudaMemset(M, 0, 256));
	

	for (int i = 0; i < 16; i++)
	{
		CHECK(cudaMalloc((uint32_t * *)& XX[i], 2048));
		CHECK(cudaMalloc((uint32_t * *)& YY[i], 2048));
		CHECK(cudaMemset(XX[i], 0, 2048));
		CHECK(cudaMemset(XX[i], 0, 2048));

		CHECK(cudaMalloc((uint8_t * *)& xx[i], 256));
		CHECK(cudaMalloc((uint8_t * *)& yy[i], 256));
		CHECK(cudaMemset(xx[i], 0, 256));
		CHECK(cudaMemset(yy[i], 0, 256));

		CHECK(cudaMalloc((uint32_t * *)& Out[i], 256));
		CHECK(cudaMemset(Out[i], 0, 256));

		cudaStreamCreate(&stream[i]);
		CHECK(cudaMalloc((int**)& flag[i], 4))
	}
	
	//R = 2^1023 nbits = 1024
	char str1[] = "79eec1e33a41bf4592557bb1991b1830d4b445f55e3c9e683afc7a7f4abf05549a5e7ea811f8c3faf58450c2eafce1a25c5eb49821d0f930247ef2c6a6e426f01f91a6090292a433d84b93a1e6c5ba933c48f48923aa727f3de18c5fa4f1c0f7cce43cf407f94ee1d316d572b4428c7399158b76fa15f8b3dfbb36bd5f4bc5d1";
	bignum_from_string(nn, str1, 256);
	char str2[] = "233c05371e4c85731b382c88438ffacb918b8e73bb099554d546c43728684ea805fbac69f0d78bfa671c17225c393b1269d2cc28f20cab1568566edd4cb8bd2f59e4b25f4b3787af54e002216bc42a34a2bdbd7bfe4ddab35dde5256fc7bfbc1b39f641c86e99950768214e69b18f806b0d200908484eb7cf6e817ab57400861";
	bignum_from_string(d, str2, 256);
	char str3[] = "4e29e645da6efddda068a8dcfceea970a5e86f7b518655cd3fba103d6899618a6b7caa86df16f28f7bdadbe2ad250794c9f20c9c42338624ab077f9f9ae3733a5c3bf8b4686b56cfe635be0010bf734fdc2a4f2ce5cf920fd4e79c6b7330a8fc2025e61d33dd8b3056390a2226d9d9eaec37f7aea1682f25120c260ecb165823";
	bignum_from_string(val, str3, 256);
	char r2ms[] = "1a32ca1d9343f9ac08567501d91b0b29540e5e6914aaf46c460b92007b6264ca7a4be15e5346933dd2865022a2535729ea817c215f80714384b8235705b88bc3a295fe00ae789bd241d5816e5d617c362a2ed1bdd8b45ca26f558a987de829afe0253c33b6a7bab59c35429c29c4ab63a0ab16c7f8c4b9319f6f1947266522a5";
	bignum_from_string(r2m, r2ms, 256);
	char str4[] = "3de12848fef2d6ddfaff968d0baf084bb5298ef79a9d6b5d8d36dc8d91d21778cd2b0258797ebe7662c5c5167d8ada581ff567183a54e2d8d9e63d51b8e95b66723a1345c434cba4e7f12ea1d5aa66eeb1d44be48b3779ba4b42537da959cc019a313c3950de5e809860a3f32e6214f445c56a8fb5c8ccb60d06b3d2b6314ccf";
	bignum_from_string(_n, str4, 256);
	e[0] = 65537;

	/*
	//R = 2^511 nbits = 512
	char str1[] = "758463d46999c11496449db8dddd1e407de2e9a8f33612f454866acddd759da8173d4e3fe8c4eaf121f86f87ac8e1d58f54e2c6a80bcf8c404884795252224ad";
	bignum_from_string(nn, str1, 128);
	char str2[] = "68827b718d1452d4e72a5085f6b14dd516df34e3ae9fb94d96da0fa3d33e651cc244b0275a24ab0753b5c01eac2f8f0d700c587bbd6d8aeb6a4e99e1a9372655";
	bignum_from_string(d, str2, 128);
	char str3[] = "45462476f31c3dfde5ac5fde4862d33d917f52255d80555b543584a32b71762a1fc719a341c0e925e9fff02a657764ae78b143d324cfc8892695c55801237885";
	bignum_from_string(val, str3, 128);
	char r2ms[] = "47395beb0ae85106f9f8548040a9b165d9a37499d0d98a14a5bcd0b943d0549be18b2ced65bfc42db40331f3ec67faf9cccf19e51d3ef7a09e03ebb1855d5e5e";
	bignum_from_string(r2m, r2ms, 128);
	char str4[] = "6629e97280ecccca1530ec12f59413fb353ccfb99a050e1203b3c9df6753928d7adfc28193416c314ff21d8c00e17814d09dccfa2850d057ceef739ce5a920db";
	bignum_from_string(_n, str4, 128);
	e[0] = 65537;
	*/
	//R = 2^12 nbits = 12
	//nn[0] = 3233;
	//e[0] = 17;
	//d[0] = 2753;
	//r2m[0] = 1179;
	//val[0] = 855;
	//_n[0] = 2207;

	ebits = bignum_numbits(e);
	dbits = bignum_numbits(d);
	nbits = bignum_numbits(nn);

	int limit = 1;
	int n = nbits / 8, m = nbits / 8, L = -1;
	for (; limit <= n + m; limit <<= 1, L++);
	CHECK(cudaMalloc((int**)& d_rev, limit * sizeof(int)));
	CHECK(cudaMalloc((ll * *)& d_root, limit * sizeof(ll)));

	for (int i = 0; i < limit; i++)
		rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << L);
	CHECK(cudaMemcpy(d_rev, &rev, limit * sizeof(int), cudaMemcpyHostToDevice));

	inv = power(limit, P - 2);
	ll temp_w = power(G, (P - 1) / limit);
	root[0] = 1;
	for (int i = 1; i < limit; ++i)
	{
		root[i] = root[i - 1] * temp_w % P;
	}
	CHECK(cudaMemcpy(d_root, root, limit * sizeof(ll), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(x, val, limit / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(y, r2m, limit / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(M, _n, limit / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	NumToNtt((uint8_t*)M, _n_ntt, d_rev, d_root, limit);
	CHECK(cudaMemcpy(M, nn, limit / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	NumToNtt((uint8_t*)M, n_ntt, d_rev, d_root, limit);
	CHECK(cudaDeviceSynchronize());
	return limit;
}

void freemem()
{
	cudaFreeHost(nn);
	cudaFreeHost(e);
	cudaFreeHost(d);
	cudaFreeHost(r2m);
	cudaFreeHost(_n);
	cudaFreeHost(val);

	cudaFree(n_ntt);
	cudaFree(_n_ntt);
	cudaFree(x);
	cudaFree(y);
	cudaFree(M);

	for (int i = 0; i < 16; i++)
	{
		cudaFree(XX[i]);
		cudaFree(YY[i]);
		cudaFree(xx[i]);
		cudaFree(yy[i]);
		cudaFree(Out[i]);
		cudaFree(flag[i]);
		cudaStreamDestroy(stream[i]);
	}

	cudaFree(d_rev);
	cudaFree(d_root);

}

int main()
{
	double iStart, iElaps;
	int limit = 1;
	iStart = seconds();
	limit = initongpu();
	iElaps = seconds() - iStart;
	cout << "Initial time elapsed" << iElaps << "sec" << endl;

	iStart = seconds();
	for (int i = 0; i < 16; i++)
	{
		MonExpOnGPU(x, d, limit, dbits, i);
	}
	iElaps = seconds() - iStart;
	cout << "GPU encrypt time elapsed" << iElaps << "sec" << endl;


	iStart = seconds();
	for (int i = 0; i < 16; i++)
	{
		cudaStreamSynchronize(stream[i]);
		MonExpOnGPU((uint8_t*)Out[i], e, limit, ebits, i);
	}
	iElaps = seconds() - iStart;
	cout << "GPU decrypt time elapsed" << iElaps << "sec" << endl;

	for (int i = 0; i < 16; i++)
	{
		PrintOnGpu << <1, limit / 2, 0, stream[i] >> > (Out[i]);
		cudaStreamSynchronize(stream[i]);
		printf("\n\n");
	}
	
	freemem();
}