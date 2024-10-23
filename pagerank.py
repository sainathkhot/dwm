import numpy as np

# Step 1: Create the adjacency matrix for the web graph
# Example: 4 pages (A, B, C, D) and links between them
adjacency_matrix = np.array([[0, 0, 1, 0],  # A links to C
                             [1, 0, 0, 1],  # B links to A and D
                             [1, 0, 0, 0],  # C links to A
                             [1, 0, 1, 0]]) # D links to A and C

# Step 2: Define constants
n = adjacency_matrix.shape[0]   # Number of pages
d = 0.85  # Damping factor (probability of following a link)
epsilon = 1e-6  # Convergence threshold
max_iterations = 100  # Maximum iterations to run

# Step 3: Initialize PageRank vector
pagerank = np.ones(n) / n  # Initial rank for each page is 1/n

# Step 4: Compute the stochastic matrix from the adjacency matrix
outgoing_links = np.sum(adjacency_matrix, axis=0)  # Count of outgoing links for each page
stochastic_matrix = adjacency_matrix / outgoing_links  # Transition probabilities
stochastic_matrix = np.nan_to_num(stochastic_matrix)  # Handle division by zero

# Step 5: Iterative calculation of PageRank
for i in range(max_iterations):
    new_pagerank = (1 - d) / n + d * np.dot(stochastic_matrix, pagerank)  # PageRank formula
    if np.linalg.norm(new_pagerank - pagerank, 2) < epsilon:  # Check convergence
        print(f"Converged after {i} iterations.")
        break
    pagerank = new_pagerank

# Step 6: Output the final PageRank values
print("Final PageRank values:")
for i, rank in enumerate(pagerank):
    print(f"Page {chr(65+i)}: {rank}")
