import network as nx
import matplotlib.pyplot as plt

ds_size= 500
exps = 5
n=100

fig, axes = plt.subplots(exps,ds_size, figsize=(n,n))
for i in range(ds_size):
    g = nx.connected_watts_strogatz_graph(n, n/10, 0.2)
    adj_g = nx.to_numpy_array (g)

    