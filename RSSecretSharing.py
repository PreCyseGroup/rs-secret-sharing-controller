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
        return pow(a, -1, self.PRIME)

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
        Q = [0] * (len(A))
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
            if self.deg(R1) < max_degree + max_error_count:
                G, leftover = self.poly_divmod(R1, T1)
                if leftover == []:
                    return G, T1
                return None
            R0, S0, T0, R1, S1, T1 = R1, S1, T1, R2, self.poly_sub(S0, self.poly_mul(S1, Q)), self.poly_sub(T0, self.poly_mul(T1, Q))

    def shamir_share(self, secret):
        secret = int(secret)
        polynomial = [secret] + [random.randrange(self.PRIME) for _ in range(self.T)]
        shares_of_secret = [self.poly_eval(polynomial, p) for p in self.POINTS]
        print (f"Shares of the Original Data: {shares_of_secret}")
        return(shares_of_secret)
    
    def shamir_decode(self, shares):
        valid_shares = [(p, v) for p, v in zip(self.POINTS, shares) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        min_shares_needed = self.T + 1
        if len(valid_shares) < min_shares_needed:
            raise ValueError(f"Not enough valid shares. Need at least {min_shares_needed} for polynomial of degree {self.T}, but got {len(valid_shares)}")
        points, values = zip(*valid_shares[:min_shares_needed])
        polynomial = self.lagrange_interpolation(points, values)
        secret = self.poly_eval(polynomial, 0)
        return secret
    
    def shamir_robust_reconstruct(self, shares):
        def is_nan(value):
            return isinstance(value, float) and np.isnan(value)
        # Find indices of NaN values
        nan_indices = [i for i, v in enumerate(shares) if v is None or is_nan(v)]
        points_values = [(p, v) for p, v in zip(self.POINTS, shares) if v is not None and not is_nan(v)]
        assert len(points_values) >= self.N - self.MAX_MISSING
        points, values = zip(*points_values)
        polynomial, error_locator = self.gao_decoding(points, values, self.R, self.MAX_MANIPULATED)
        if polynomial is None:
            raise Exception("Too many errors, cannot reconstruct")
        secret = self.poly_eval(polynomial, 0)
        error_indices = [i for i, v in enumerate(self.poly_eval(error_locator, p) for p in self.POINTS) if v == 0]
        
        # Handle potential IndexError
        error_index = error_indices[0] if error_indices else None
        nan_index = nan_indices[0] if nan_indices else None
        
        # print(f"Reconstructed Secret: {secret}")
        # print(f"Error Indices: {error_index}")
        # print(f"NaN Indices: {nan_index}")
        return secret, error_index, nan_index

    def shares_noissy_channel(self, shares, seed=None): 
        if isinstance(shares, np.ndarray):
            is_array = True
            if shares.ndim == 1:
                shares = shares.reshape(1, -1)
            rows, cols = shares.shape
        elif isinstance(shares, list):
            is_array = False
            if all(isinstance(elem, list) for elem in shares):
                rows = len(shares)
                cols = len(shares[0])
            else:
                shares = [shares]
                rows = 1
                cols = len(shares[0])
        else:
            raise TypeError("Unsupported data type for shares")
        if self.MAX_MANIPULATED + self.MAX_MISSING > cols:
            raise ValueError("The total of manipulated and missing cannot be greater than the number of columns.")
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
        if is_array:
            modified_matrix = shares.copy()
            if modified_matrix.dtype != object:
                modified_matrix = modified_matrix.astype(object)
        else:
            modified_matrix = [row.copy() for row in shares]
        columns_to_modify = random.sample(range(cols), self.MAX_MANIPULATED + self.MAX_MISSING)
        columns_to_replace_with_random = columns_to_modify[:self.MAX_MANIPULATED]
        columns_to_replace_with_nan = columns_to_modify[self.MAX_MANIPULATED:]
        if is_array:
            for col in columns_to_replace_with_random:
                modified_matrix[:, col] = np.array([random.randrange(self.PRIME) for _ in range(rows)], dtype=object)
        else:
            for col in columns_to_replace_with_random:
                for row in range(rows):
                    modified_matrix[row][col] = random.randrange(self.PRIME)
        if is_array:
            for col in columns_to_replace_with_nan:
                modified_matrix[:, col] = np.array([float('nan')] * rows, dtype=object)
        else:
            for col in columns_to_replace_with_nan:
                for row in range(rows):
                    modified_matrix[row][col] = float('nan')
        if rows == 1:
            if is_array:
                print (f"Modified Matrix: {modified_matrix[0]}")
                return modified_matrix[0]
            else:
                print (f"Modified Matrix: {modified_matrix[0]}")
                return modified_matrix[0]
        else:
            print (f"Modified Matrix: {modified_matrix}")
            return modified_matrix

    def shares_of_vector(self, vector):
        share_matrix = np.zeros((len(vector), self.N), dtype=object)
        for index, state in enumerate(vector):
            share_matrix[index, :] = self.shamir_share(state)
        share_matrix_noisy = self.shares_noissy_channel(share_matrix, self.RAN_SEED)
        return share_matrix_noisy  # Return the noisy shares
    
    def simulate_missing_data(self, shares, columns_to_replace_with_nan):
        if isinstance(shares, np.ndarray):
            is_array = True
            if shares.ndim == 1:
                shares = shares.reshape(1, -1)
            rows, cols = shares.shape
        elif isinstance(shares, list):
            is_array = False
            if all(isinstance(elem, list) for elem in shares):
                rows = len(shares)
                cols = len(shares[0])
            else:
                shares = [shares]
                rows = 1
                cols = len(shares[0])
        else:
            raise TypeError("Unsupported data type for shares")
        for col in columns_to_replace_with_nan:
            if col < 0 or col >= cols:
                raise ValueError(f"Column index {col} is out of bounds.")
        for col in columns_to_replace_with_nan:
            if is_array:
                shares[:, col] = np.array([float('nan')] * rows, dtype=object)
            else:
                for row in range(rows):
                    shares[row][col] = float('nan')
        return shares

    def simulate_manipulated_data(self, shares, columns_to_replace_with_random):
        if isinstance(shares, np.ndarray):
            is_array = True
            if shares.ndim == 1:
                shares = shares.reshape(1, -1)
            rows, cols = shares.shape
        elif isinstance(shares, list):
            is_array = False
            if all(isinstance(elem, list) for elem in shares):
                rows = len(shares)
                cols = len(shares[0])
            else:
                shares = [shares]
                rows = 1
                cols = len(shares[0])
        else:
            raise TypeError("Unsupported data type for shares")
        for col in columns_to_replace_with_random:
            if col < 0 or col >= cols:
                raise ValueError(f"Column index {col} is out of bounds.")
        for col in columns_to_replace_with_random:
            if is_array:
                shares[:, col] = np.array([random.randrange(self.PRIME) for _ in range(rows)], dtype=object)
            else:
                for row in range(rows):
                    shares[row][col] = random.randrange(self.PRIME)
        return shares

# # Example 128-bit prime number
# PRIME = 340282366920938463463374607431768211507  # This is 2^128 - 159
# # Initialize the RSSecretSharing class
# rs = RSSecretSharing(PRIME=PRIME, K=1, N=5, T=1, MAX_MISSING=1, MAX_MANIPULATED=1, RAN_SEED=42)
# # Secret to share
# secret = 5
# # Generate shares
# shares = rs.shamir_share(secret)
# print("Shares:", shares)
# shamir_decode = rs.shamir_decode(shares)
# print("Shamir Decode:", shamir_decode)
# # Simulate a noisy channel
# noisy_shares = rs.shares_noissy_channel(shares)
# print("Noisy Shares:", noisy_shares)
# print("Shamir Decode for Noisy Shares:", rs.shamir_decode(noisy_shares))
# # Reconstruct the secret
# reconstructed_secret, error_indices, _ = rs.shamir_robust_reconstruct(noisy_shares)
# print("Reconstructed Secret:", reconstructed_secret)
# print("Error Indices:", error_indices)