# Nombre: Cruz Eduardo Gamez Rodriguez
# Registro: 21110301

import matplotlib.pyplot as plt  # Biblioteca para graficar.
import networkx as nx  # Biblioteca para trabajar con grafos.

# Clase Union-Find (Disjoint Set Union) para gestionar conjuntos disjuntos.
class UnionFind:
    def __init__(self, n):
        # Inicializa cada nodo como su propio padre y asigna rango 0 a todos.
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        # Busca la raíz del conjunto al que pertenece el nodo u.
        # Aplica compresión de caminos para optimizar futuras búsquedas.
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Compresión
        return self.parent[u]

    def union(self, u, v):
        # Une los conjuntos a los que pertenecen u y v.
        # Usa el rango para mantener el árbol lo más plano posible.
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# Algoritmo de Kruskal para construir el árbol de expansión mínimo o máximo.
def kruskal_simulator(graph_edges, num_nodes, max_tree=False):
    # Ordena las aristas por peso.
    # Si max_tree=True, ordena de mayor a menor (para árbol de máximo costo).
    graph_edges = sorted(graph_edges, key=lambda x: x[2], reverse=max_tree)
    uf = UnionFind(num_nodes)  # Inicializa la estructura Union-Find.
    mst_edges = []  # Lista para almacenar las aristas del árbol resultante.
    total_cost = 0  # Acumulador del costo total.

    print(f"\nConstrucción del Árbol de {'Máximo' if max_tree else 'Mínimo'} Costo usando Kruskal:")

    # Recorre todas las aristas ordenadas.
    for u, v, weight in graph_edges:
        # Si los nodos u y v no están ya conectados en el mismo conjunto,
        # se puede añadir la arista sin crear ciclos.
        if uf.find(u) != uf.find(v):
            uf.union(u, v)  # Une los conjuntos de u y v.
            mst_edges.append((u, v, weight))  # Agrega la arista al árbol.
            total_cost += weight  # Suma el peso al costo total.
            print(f"Añadiendo arista ({u}, {v}) con peso {weight}")

            # Si ya se tienen n-1 aristas, se ha completado el árbol.
            if len(mst_edges) == num_nodes - 1:
                break

    # Muestra el costo total final del árbol.
    print(f"\nCosto total del Árbol de {'Máximo' if max_tree else 'Mínimo'} Costo: {total_cost}")
    return mst_edges, total_cost  # Devuelve las aristas del árbol y su costo.

# Función para graficar el árbol generado.
def draw_graph(edges, num_nodes, title):
    G = nx.Graph()  # Crea un grafo vacío.
    G.add_nodes_from(range(num_nodes))  # Añade los nodos numerados del 0 al num_nodes-1.
    G.add_weighted_edges_from(edges)  # Añade las aristas con sus respectivos pesos.

    pos = nx.spring_layout(G)  # Calcula automáticamente posiciones para los nodos.
    plt.figure(figsize=(8, 6))  # Tamaño del gráfico.
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)  # Dibuja el grafo.
    
    # Crea etiquetas con los pesos de las aristas.
    edge_labels = {(u, v): f"{w}" for u, v, w in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")  # Dibuja las etiquetas.

    plt.title(title)  # Título del gráfico.
    plt.show()  # Muestra el gráfico.

# -----------------------
# Ejecución del programa
# -----------------------

# Lista de aristas del grafo con sus respectivos pesos.
# Representa conexiones entre oficinas y el costo de conectar cada par.
edges = [
    (0, 1, 4), (0, 2, 3), (1, 2, 1), (1, 3, 2),
    (2, 3, 4), (3, 4, 2), (4, 5, 6), (3, 5, 3)
]
num_nodes = 6  # Número total de nodos/oficinas.

# Árbol de Mínimo Costo
print("=== Árbol de Mínimo Costo ===")
mst_min_edges, min_cost = kruskal_simulator(edges, num_nodes, max_tree=False)
draw_graph(mst_min_edges, num_nodes, "Árbol de Mínimo Costo - Conexión de Oficinas")

# Árbol de Máximo Costo
print("\n=== Árbol de Máximo Costo ===")
mst_max_edges, max_cost = kruskal_simulator(edges, num_nodes, max_tree=True)
draw_graph(mst_max_edges, num_nodes, "Árbol de Máximo Costo - Conexión de Oficinas")
