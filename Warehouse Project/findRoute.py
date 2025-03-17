import heapq


#---------Tìm đường đi trong map 2d dùng A*--------#
# Hướng di chuyển 4 chiều: trái, phải, lên, xuống
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def heuristic(a, b):
    """Hàm heuristic dùng khoảng cách Manhattan"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f, (x, y))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            total_distance = 0
            prev = None
            
            while current in came_from:
                path.append(current)
                if prev:
                    total_distance += 3  # Mỗi bước di chuyển có chi phí 3
                prev = current
                current = came_from[current]
            
            path.append(start)
            path.reverse()
            return path, total_distance  # Trả về cả đường đi và độ dài quãng đường
        
        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Cho phép đi vào start và goal, nhưng các ô # khác bị chặn
                if grid[neighbor[0]][neighbor[1]] == 0 and neighbor not in [start, goal]:
                    continue  # Bỏ qua vật cản trừ start và goal
                
                temp_g_score = g_score[current] + 1  # Di chuyển ngang/dọc có chi phí 1
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, float('inf')  # Không tìm thấy đường đi

'''
grid = [[3, 3, 3, 3, 3, 3], [3, 2, 3, 3, 2, 3], [3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3], [3, 2, 3, 3, 2, 3], [3, 3, 3, 3, 3, 3]]
start = (1,1)  # Bắt đầu từ một ô #
goal = (1,1)   # Kết thúc ở một ô #

path, distance = astar(grid, start, goal)

if path:
    print("Đường đi tìm được:", path)
    print("Độ dài quãng đường:", distance)
else:
    print("Không có đường đi.")
'''