import random
import numpy as np
import heapq
import time
import csv
import copy

#dùng A* để tìm đường đi trong map 2d có vật cản
import findRoute
from findRoute import astar


# dữ liệu kho hàng {(shelve:layer) : [productName, quantity]}
warehouse = {}
demand = {}
robot_capacity = 0
robot_maxDistance = 0

#đọc dữ liệu từ file
with open("warehouse2.txt", "r") as file:
    lines = file.readlines()
    num_shelves = int(lines[0].split(':')[1].strip())
    num_layers = int(lines[1].split(':')[1].strip())
    NumberOfCells = int(lines[2].split(':')[1].strip())
    num_required_products = int(lines[3].split(':')[1].strip())
    robot_capacity = int(lines[4].split(':')[1].strip())
    robot_maxDistance = int(lines[5].split(':')[1].strip())

    shelf = 1
    layer = 1

    for i in range(num_shelves * num_layers):
        parts = lines[6 + i].strip().split()
        product = parts[0]
        quantity = int(parts[1])
        warehouse[(shelf, layer)] = [product, quantity]

        # Cập nhật vị trí
        layer += 1
        if layer > num_layers:
            layer = 1
            shelf += 1

    # Lưu yêu cầu đơn hàng
    for i in range(num_required_products):
        parts = lines[6 + num_shelves * num_layers + i].strip().split()
        product = parts[0]
        quantity = int(parts[1])
        demand[product] = quantity


#kết nối với layout kho hàng
shelves_position = {
    0: (5, 2),  #vị trí trả hàng
    1: (1, 1),  # Kệ 1 (Góc trên trái)
    2: (1, 4),  # Kệ 2 (Góc trên phải)
    3: (4, 1),  # Kệ 3 (Góc dưới trái)
    4: (4, 4)   # Kệ 4 (Góc dưới phải)
}
#wareMap : 0 là vị trí kệ, 3 là đường đi
wareMap = [[3]*6 for _ in range(6)]
for k,(x,y) in shelves_position.items():
    if k != 0:
        wareMap[x][y] = 0

#tọa độ các kệ
coord = np.array([v for k, v in shelves_position.items() if k > 0]) #lấy tọa độ các kệ hàng (key>0)
key = [i for i in shelves_position.keys() if i > 0]

shelve_struct = {}
for sp in shelves_position.keys():
    if sp != 0:
        shelve_struct[sp] = [sl for sl in warehouse.keys() if sl[0] == sp]

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1) #chỉ tính khoảng cách theo 4 hướng, không theo đường chéo

# hàm phân cụm k-means
def kmeans(X, k, max_iters=100, tol=1e-4):
    '''
    X : tọa độ các điểm
    '''
    np.random.seed(0) # cố định kết quả qua những lần random

    #chọn k điểm ngẫu nhiên làm trung tâm
    #X.shape[0] : số hàng(ma trận) hay số điểm dữ liệu
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]


    for _ in range(max_iters):
        '''
        gán nhãn các điểm theo gốc gần nhất
        từng tọa độ - các tọa độ gốc;
        gốc 0 -> k-1 tạo khoảng cách d0 -> d(k-1)
        np.argmin(axis = None) : lấy chỉ số nhỏ nhất ở mảng 1 chiều
        labels : dãy gán điểm cho gốc [0,3,k-1,...] // điểm 1 nhãn 1, điểm 2 nhãn 3...
        '''
        labels = np.array([np.argmin(euclidean_distance(x, centroids)) for x in X])

        #tính lại tọa độ gốc
        '''
        --> lấy tọa độ trung bình các điểm đã được gán theo nhãn
        labels == i : tập các điểm có nhãn i
        mean : lấy trung bình
        axis=0 : lấy theo cột // cột x, cột y, cột z
        '''
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        #sai số thỏa mãn thì dừng
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids



# ------------------GA--------------------#
'''
cá thể là thứ tự thăm (shelve : layer) -> (shelve : layer)
'''
#khởi tạo quần thể
def initial_population(size, eg_member):
    """Tạo quần thể ban đầu"""
    return [random.sample(eg_member, len(eg_member)) for _ in range(size)]

def calculate_distance(path):
    total_d = 0
    start = shelves_position[path[0][0]]
    high = 0 #độ cao hiện tại
    for i in range(1,len(path)):
        goal = shelves_position[path[i][0]] # lấy tọa độ xy của member
        _,d = astar(wareMap, start, goal)
            
        #quãng đường theo mặt phẳng + chênh lệch độ cao
        total_d = total_d + d + abs(high - path[i][1])

        #cập nhật start,high
        start = goal
        high = path[i][1]

    return total_d

def calculate_fitness(population):
    global warehouse, demand
    fit = []
    for individual in population: 
        total_distance = 0
        load = 0
        collected = {item: 0 for item in demand}
        path = [(0, 0)]  # Robot bắt đầu từ kho
        
        for shelf, layer in individual:
            product, quantity = warehouse[(shelf, layer)]
            
            if collected[product] < demand[product]:
                needed = min(demand[product] - collected[product], quantity, robot_capacity - load)
                collected[product] += needed
                load += needed
                warehouse[(shelf, layer)][1] -= needed
                path.append((shelf, layer))

                if load >= robot_capacity:  # Nếu đầy tải, quay về kho
                    total_distance += calculate_distance(path) + calculate_distance([(0, 0), path[-1]])
                    path = [(0, 0)]
                    load = 0  

                if needed < quantity:  # Nếu còn hàng, chèn lại vào danh sách
                    individual.append((shelf, layer))

            if sum(collected.values()) >= sum(demand.values()):
                break  

            if total_distance + calculate_distance(path) > robot_maxDistance:
                total_distance += calculate_distance(path) + calculate_distance([(0, 0), path[-1]])
                path = [(0, 0)]
        
        if path[-1] != (0, 0):
            total_distance += calculate_distance(path) + calculate_distance([(0, 0), path[-1]])
        
        fitness = 100/ total_distance
        fit.append(fitness)  
    return fit

# Chọn cha mẹ (Tournament Selection)
def find_parent(population, fitness, k=5):
    selected = random.sample(list(zip(population, fitness)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

# Lai ghép (Crossover - Order Crossover)
def hybrid(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    # Sao chép đoạn giữa từ parent1
    child[start:end + 1] = parent1[start:end + 1]
    # Điền phần còn lại từ parent2
    current_index = 0
    for gene in parent2:
        if gene not in child:
            while child[current_index] != -1:
                current_index += 1
            child[current_index] = gene

    return child

# Đột biến ()
def mutate(child,MUTATION_RATE = 0.1):
    if random.random() < MUTATION_RATE:
        idx1, idx2 = np.random.choice(len(child), 2, replace=False)  # Chọn 2 vị trí khác nhau
        child[idx1], child[idx2] = child[idx2], child[idx1]  # Hoán đổi hai giá trị
    return child


# Tạo quần thể mới
def generate_New_population(population, fitness):
    new_population = []
    for _ in range(len(population) // 2):
        parent1 = find_parent(population, fitness)
        parent2 = find_parent(population, fitness)

        child1 = mutate(hybrid(parent1, parent2))
        child2 = mutate(hybrid(parent2, parent1))
        new_population.extend([child1, child2])
    return new_population



#---------Chạy Kmeans và GA------------------#

file_path = "Kmeans_GA_result.csv"
file_name = r"C:\Users\admin\OneDrive\Máy tính\Warehouse Project\by_KmeanN_GA.py"

with open(file_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter="\t")
    columns = ["Algorithm","No.Batch","PopulationSize", "MaxGeneration", "MutationRate", "FileName",
               "Generation","Total Orders","BestFitness", "ExecutingTime", "BestDistance"]
    writer.writerow(columns)

    start_time = time.time()

    #phân cụm
    num_clus = 2
    labels,_ = kmeans(coord,num_clus)
      

    #gom nhóm theo nhãn
    batchs = []
    #cluster_indices = [np.where(labels == i)[0] for i in range(k)]
    cluster_keys = [[key[i] for i in np.where(labels == j)[0]] for j in range(num_clus)]
    for i, k in enumerate(cluster_keys):
        batch = []
        for j in k:
            #extend : thêm tất cả phần tử của iterable(danh sách, tuple, set, v.v.) vào cuối danh sách hiện có.
            batch.extend(shelve_struct[j])
        batchs.append(batch)


    total_distance = 0
    #Dùng GA cho mỗi nhóm :
    for i in range(num_clus):
        NUM_GENERATION = 60
        POPULATION_SIZE = 40
        population = initial_population(POPULATION_SIZE,batchs[i])

        best_distance = float('inf')
        best_route = []

        # Vòng lặp tiến hóa
        for ng in range(NUM_GENERATION):
            fitness = calculate_fitness(population)
            best_fitness = max(fitness)

            best = fitness.index(best_fitness)
            min_distance = 100/best_fitness

            if min_distance < best_distance:
                best_distance = min_distance
                best_route = population[best]

            population = generate_New_population(population, fitness)

            execute_time = time.time() - start_time

            # Ghi kết quả vào file
            new_row = ["Kmeans + GA", i, POPULATION_SIZE, NUM_GENERATION,0.1, file_name, ng,
                    len(batchs[i]), best_fitness, execute_time, best_distance]
            #writer.writerow(new_row)
        
        total_distance += best_distance
        
        print("Nhóm",i,':')
        #print(centre[i])
        print('Thứ tự lấy hàng :',best_route)
        print('Quãng đường vận chuyển tốt nhất tìm được :',best_distance)
    print('Total distance',total_distance)

'''
{(1, 1): ['Abiu', 9],
(1, 2): ['Abiu', 14], 
(1, 3): ['Abiu', 13],
(2, 1): ['Acerola', 19], 
(2, 2): ['Acerola', 3], 
(2, 3): ['Acerola', 13], 
(3, 1): ['Acerola', 18], 
(3, 2): ['Acerola', 1], 
(3, 3): ['Acerola', 14], 
(4, 1): ['Acerola', 18], 
(4, 2): ['Acerola', 4], 
(4, 3): ['Acerola', 5]}

demand
{'Abiu': 32, 'Acerola': 25}
'''
