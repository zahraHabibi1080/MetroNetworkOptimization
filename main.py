import pandas as pd
import numpy as np
import ast
import random

# ==========================================
# بخش 1: بارگذاری و آماده‌سازی داده‌ها
# ==========================================
def load_and_clean_data(file_path):
    data = pd.read_excel(file_path)
    
    # پیدا کردن ستون‌های فاصله
    distance_cols = [col for col in data.columns if "فاصله با ایستگاه" in col]
    rename_dict = {col: f"dist{i+1}" for i, col in enumerate(distance_cols)}
    data.rename(columns=rename_dict, inplace=True)

    # تابع تبدیل متن به لیست
    def safe_eval(val):
        if pd.isna(val) or val == "" or val == "[]":
            return []
        try:
            return ast.literal_eval(str(val))
        except:
            return []

    # اعمال روی ستون‌های فاصله
    for col in rename_dict.values():
        data[col] = data[col].apply(safe_eval)
        
    return data, rename_dict

def build_graph(df, rename_dict):
    num_stations = int(df['ID'].max() + 1)
    adj_matrix = np.full((num_stations, num_stations), np.inf)
    np.fill_diagonal(adj_matrix, 0)
    
    id_to_name = pd.Series(df['اسم ایستگاه'].values, index=df['ID']).to_dict()
    
    for _, row in df.iterrows():
        u = int(row['ID'])
        for col in rename_dict.values():
            val = row[col]
            if isinstance(val, list) and len(val) == 2:
                v, weight = int(val[0]), float(val[1])
                adj_matrix[u][v] = weight
                adj_matrix[v][u] = weight  # شبکه دوطرفه
    return adj_matrix, id_to_name

# ==========================================
# بخش 2: الگوریتم کلونی مورچگان (ACO)
# ==========================================
class MetroAntColony:
    def __init__(self, adj_matrix, start, end, n_ants=30, iterations=100, alpha=1.0, beta=2.0, evaporation=0.5):
        self.adj_matrix = adj_matrix
        self.start = start
        self.end = end
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha  # اهمیت فرومون
        self.beta = beta    # اهمیت فاصله (ابتکار)
        self.evaporation = evaporation
        self.pheromones = np.ones(adj_matrix.shape)

    def run(self):
        best_path = None
        best_dist = np.inf

        for _ in range(self.iterations):
            paths = self._build_all_paths()
            self._update_pheromones(paths)
            
            for path, dist in paths:
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
        return best_path, best_dist

    def _build_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = [self.start]
            visited = {self.start}
            while path[-1] != self.end:
                current = path[-1]
                next_node = self._select_next(current, visited)
                if next_node is None: break
                path.append(next_node)
                visited.add(next_node)
                if len(path) > 100: break # جلوگیری از چرخه طولانی
            
            if path[-1] == self.end:
                dist = sum(self.adj_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
                all_paths.append((path, dist))
        return all_paths

    def _select_next(self, current, visited):
        probabilities = []
        for next_node, dist in enumerate(self.adj_matrix[current]):
            if dist != np.inf and dist > 0 and next_node not in visited:
                tau = self.pheromones[current][next_node] ** self.alpha
                eta = (1.0 / dist) ** self.beta
                probabilities.append((next_node, tau * eta))
        
        if not probabilities: return None
        nodes, weights = zip(*probabilities)
        return random.choices(nodes, weights=weights)[0]

    def _update_pheromones(self, paths):
        self.pheromones *= (1 - self.evaporation)
        for path, dist in paths:
            for i in range(len(path)-1):
                self.pheromones[path[i]][path[i+1]] += (1.0 / dist)

# ==========================================
# بخش 3: اجرای اصلی و خروجی
# ==========================================
if __name__ == "__main__":
    file_path = "metro.xlsx" # فایل را در کنار کد قرار دهید
    
    # 1. آماده سازی داده
    data, rename_dict = load_and_clean_data(file_path)
    adj_matrix, id_to_name = build_graph(data, rename_dict)
    
    # 2. دریافت ورودی (مثلاً از تجریش ID:1 به تئاتر شهر ID:67)
    start_node = 1
    end_node = 67
    
    print(f"Searching for the best path from {id_to_name[start_node]} to {id_to_name[end_node]}...")
    
    # 3. اجرا
    aco = MetroAntColony(adj_matrix, start_node, end_node)
    best_path, total_dist = aco.run()
    
    # 4. نمایش نتیجه
    if best_path:
        path_names = [id_to_name[node] for node in best_path]
        print("\n✅ Shortest Path Found:")
        print(" -> ".join(path_names))
        print(f"\n📏 Total Distance: {total_dist:.2f} km")
    else:
        print("❌ No path found.")
        