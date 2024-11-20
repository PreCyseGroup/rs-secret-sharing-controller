from copy import copy, deepcopy
import random
import numpy as np

class RSSecretSharing:
    def __init__(self, PRIME, K=1, N=5, T=1, MAX_MISSING=1, MAX_MANIPULATED=1, RAN_SEED=None):
        self.PRIME = PRIME  # Base field arithmetic (prime number)
        self.K = K  # Threshold for reconstruction
        self.N = N  # Total number of shares
        self.T = T  # Degree of polynomial
        self.R = T + K  # Minimum required shares for reconstruction
        self.MAX_MISSING = MAX_MISSING  # Maximum missing shares allowed
        self.MAX_MANIPULATED = MAX_MANIPULATED  # Maximum manipulated shares allowed
        self.POINTS = [p for p in range(1, N + 1)]
        self.RAN_SEED = RAN_SEED
        assert self.R + self.MAX_MISSING + 2 * self.MAX_MANIPULATED <= N

    def base_egcd(self, a, b):
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1
        while r1 != 0:
            q, r2 = divmod(r0, r1)
            r0, s0, t0, r1, s1, t1 = r1, s1, t1, r2, s0 - s1 * q, t0 - t1 * q
        return r0, s0, t0

    def base_inverse(self, a):
        _, b, _ = self.base_egcd(a, self.PRIME)
        return b if b >= 0 else b + self.PRIME

    def base_add(self, a, b):
        return (a + b) % self.PRIME

    def base_sub(self, a, b):
        return (a - b) % self.PRIME

    def base_mul(self, a, b):
        return (a * b) % self.PRIME

    def base_div(self, a, b):
        return self.base_mul(a, self.base_inverse(b))

    def expand_to_match(self, A, B):
        diff = len(A) - len(B)
        if diff > 0:
            return A, B + [0] * diff
        elif diff < 0:
            return A + [0] * abs(diff), B
        return A, B

    def canonical(self, A):
        for i in reversed(range(len(A))):
            if A[i] != 0:
                return A[:i + 1]
        return []

    def lc(self, A):
        return self.canonical(A)[-1]

    def deg(self, A):
        return len(self.canonical(A)) - 1

    def poly_add(self, A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_add(f, g) for f, g in zip(F, G)])

    def poly_sub(self, A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_sub(f, g) for f, g in zip(F, G)])

    def poly_scalarmul(self, A, b):
        return self.canonical([self.base_mul(a, b) for a in A])

    def poly_scalardiv(self, A, b):
        return self.canonical([self.base_div(a, b) for a in A])

    def poly_mul(self, A, B):
        C = [0] * (len(A) + len(B) - 1)
        for i in range(len(A)):
            for j in range(len(B)):
                C[i + j] = self.base_add(C[i + j], self.base_mul(A[i], B[j]))
        return self.canonical(C)

    def poly_divmod(self, A, B):
        t = self.base_inverse(self.lc(B))
        Q = [0] * len(A)
        R = copy(A)
        for i in reversed(range(len(A) - len(B) + 1)):
            Q[i] = self.base_mul(t, R[i + len(B) - 1])
            for j in range(len(B)):
                R[i + j] = self.base_sub(R[i + j], self.base_mul(Q[i], B[j]))
        return self.canonical(Q), self.canonical(R)

    def poly_eval(self, A, x):
        result = 0
        for coef in reversed(A):
            result = self.base_add(coef, self.base_mul(x, result))
        return result

    def lagrange_polynomials(self, xs):
        polys = []
        for i, xi in enumerate(xs):
            numerator = [1]
            denominator = 1
            for j, xj in enumerate(xs):
                if i == j:
                    continue
                numerator = self.poly_mul(numerator, [self.base_sub(0, xj), 1])
                denominator = self.base_mul(denominator, self.base_sub(xi, xj))
            poly = self.poly_scalardiv(numerator, denominator)
            polys.append(poly)
        return polys

    def lagrange_interpolation(self, xs, ys):
        ls = self.lagrange_polynomials(xs)
        poly = []
        for i in range(len(ys)):
            term = self.poly_scalarmul(ls[i], ys[i])
            poly = self.poly_add(poly, term)
        return poly

    def gao_decoding(self, points, values, max_degree, max_error_count):
        assert len(values) == len(points)
        assert len(points) >= 2 * max_error_count + max_degree
        H = self.lagrange_interpolation(points, values)
        F = [1]
        for xi in points:
            F = self.poly_mul(F, [self.base_sub(0, xi), 1])
        R0, R1 = F, H
        S0, S1 = [1], []
        T0, T1 = [], [1]
        while True:
            Q, R2 = self.poly_divmod(R0, R1)
            if self.deg(R0) < max_degree + max_error_count:
                G, leftover = self.poly_divmod(R0, T0)
                if leftover == []:
                    return G, T0
                return None
            R0, S0, T0, R1, S1, T1 = R1, S1, T1, R2, self.poly_sub(S0, self.poly_mul(S1, Q)), self.poly_sub(T0, self.poly_mul(T1, Q))

    def shamir_share(self, secret):
        polynomial = [secret] + [random.randrange(self.PRIME) for _ in range(self.T)]
        return [self.poly_eval(polynomial, p) for p in self.POINTS]

    def shamir_robust_reconstruct(self, shares):
        points_values = [(p, v) for p, v in zip(self.POINTS, shares) if v is not None]
        assert len(points_values) >= self.N - self.MAX_MISSING
        points, values = zip(*points_values)
        polynomial, error_locator = self.gao_decoding(points, values, self.R, self.MAX_MANIPULATED)
        if polynomial is None:
            raise Exception("Too many errors, cannot reconstruct")
        secret = self.poly_eval(polynomial, 0)
        error_indices = [i for i, v in enumerate(self.poly_eval(error_locator, p) for p in self.POINTS) if v == 0]
        return secret, error_indices

    def shares_noissy_channel(self, shares, seed=None): 
        if isinstance(shares,np.ndarray):
            is_array=True
            if shares.ndim==1:
                shares=shares.reshape(1,-1)
            rows,cols=shares.shape
        elif isinstance(shares,list):
            is_array=False
            if all(isinstance(elem,list) for elem in shares):
                rows=len(shares)
                cols=len(shares[0])
            else:
                shares=[shares]
                rows=1
                cols=len(shares[0])
        else:
            raise TypeError("Unsupported data type for shares")
        if self.MAX_MANIPULATED+self.MAX_MISSING>cols:
            raise ValueError("The total of manipulated and missing cannot be greater than the number of columns.")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        else:
            np.random.seed()
            random.seed()
        if is_array:
            modified_matrix=shares.copy()
            if modified_matrix.dtype!=object:
                modified_matrix=modified_matrix.astype(object)
        else:
            modified_matrix=[row.copy() for row in shares]
        columns_to_modify=random.sample(range(cols),self.MAX_MANIPULATED+self.MAX_MISSING)
        columns_to_replace_with_random=columns_to_modify[:self.MAX_MANIPULATED]
        columns_to_replace_with_nan=columns_to_modify[self.MAX_MANIPULATED:]
        if is_array:
            for col in columns_to_replace_with_random:
                modified_matrix[:,col]=np.random.randint(0,self.PRIME,size=rows)
        else:
            for col in columns_to_replace_with_random:
                for row in range(rows):
                    modified_matrix[row][col]=random.randrange(self.PRIME)
        if is_array:
            modified_matrix[:,columns_to_replace_with_nan]=float('nan')
        else:
            for col in columns_to_replace_with_nan:
                for row in range(rows):
                    modified_matrix[row][col]=float('nan')
        if rows==1:
            if is_array:
                return modified_matrix[0]
            else:
                return modified_matrix[0]
        else:
            return modified_matrix
    
    def shares_of_vector(self, vector):
        share_matrix = np.zeros((len(vector), self.N))
        share_matrix_noisy = np.zeros((len(vector), self.N))
        for index, state in enumerate(vector):
            share_matrix[index, :] = (self.shamir_share(state))
        share_matrix_noisy = self.shares_noissy_channel(share_matrix, self.RAN_SEED)
        return share_matrix_noisy
