import numpy as np
from time import time 

#Bialek paper used 1246 boids
#try nc  = 20 birds
#bialek et all calculated of 25 snapshots

start = time()
class Boid():
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.position = np.array([self.x , self.y])
        vec = (np.random.rand(2) - 0.5)*10
        self.velocity = vec

        vec = (np.random.rand(2) - 0.5)/2
        self.acceleration = vec
        self.max_force = 0.3
        self.max_speed = 5
        self.perception = 100

        self.width = width
        self.height = height



    def update(self):
        self.position += self.velocity
        self.velocity += self.acceleration
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed
        self.acceleration = np.zeros(2)


    def apply_behaviour(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)
        self.acceleration += alignment
        self.acceleration += cohesion
        self.acceleration += separation

    def edges(self):
        if self.position[0] > self.width:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = self.width

        if self.position[1] > self.height:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = self.height

    def align(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                avg_vector += boid.velocity
                total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = (avg_vector / np.linalg.norm(avg_vector)) * self.max_speed
            steering = avg_vector - self.velocity

        return steering

    def cohesion(self, boids):
        steering = np.zeros(2)
        total = 0
        center_of_mass = np.zeros(2)
        for boid in boids:
            if np.linalg.norm(boid.position - self.position) < self.perception:
                center_of_mass += boid.position
                total += 1
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = (vec_to_com / np.linalg.norm(vec_to_com)) * self.max_speed
            steering = vec_to_com - self.velocity
            if np.linalg.norm(steering)> self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

    def separation(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for boid in boids:
            distance = np.linalg.norm(boid.position - self.position)
            if self.position.all() != boid.position.all() and distance < self.perception:
                diff = self.position - boid.position
                diff /= distance
                avg_vector += diff
                total += 1
        if total > 0:
            avg_vector /= total
            if np.linalg.norm(steering) > 0:
                avg_vector = (avg_vector / np.linalg.norm(steering)) * self.max_speed
            steering = avg_vector - self.velocity
            if np.linalg.norm(steering) > self.max_force:
                steering = (steering /np.linalg.norm(steering)) * self.max_force

        return steering

#////////////////////CONTROL PARAMETERS///////////////////////////////////
width = 2000
height = 2000
boid_n = 526 #number of boids
snapshot = 50 #how many snapshots do we use to calculate the interaction parameters
time_steps = 1000
n_size =  list(range(1,5))
#////////////////////INSTANTIATE CLASSES////////////////////////////////////////
flock = [Boid(np.random.rand()*1000, np.random.rand()*1000, width, height) for _ in range(boid_n)]

#////////////////////GENERAL FUNCTIONS//////////////////////////////////////////
def distance(p1, p2):
   return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def find_nearest(pos, pos_list , n): 
    #function that returns the nearest n closest positions to a given vector. 
    idx_list = []
    dist = []
    for p in pos_list:
        dist.append(distance(pos, p))
    
    dist = np.array(dist)
    for i in range(n+1):
        idx = dist.argmin()
        dist[idx]= 99999999
        idx_list.append(idx)
        
    del idx_list[0]
    return idx_list

def update():
    global flock
    norm_vel = []
    pos = []
    posx = []
    posy = []
    velx = []
    vely = []
    for boid in flock:
        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        norm_v = boid.velocity/np.linalg.norm(boid.velocity)
        norm_vel.append(norm_v)
        pos.append(boid.position)
        posx.append(boid.position[0])
        posy.append(boid.position[1])
        velx.append(norm_v[0])
        vely.append(norm_v[1])

    #Return the velocities for each time step  
    return norm_vel, pos, posx, posy, velx, vely

def calc_c(norm_v, pos, n): 
    #pos is list of positions
    list_of_sums = []
    for i in range(boid_n):
    #find n_c closest boids
        idx_list = find_nearest(pos[i],pos,n)
        list_of_products = []
        for j in idx_list:
            a = norm_v[i]
            b = norm_v[j]
            product = np.inner(a, b)
            list_of_products.append(product)
        product_sum = np.sum(list_of_products)/n
        list_of_sums.append(product_sum)
    C = np.sum(list_of_sums)/boid_n
    return C



#////////////////////RUN SIMULATION////////////////////////

def run(time = time_steps, n_c = n_size): 
    #function to run simulation code
    #time = how many timesteps do we want to simulate for
    C_int = []
    allposx = []
    allposy = []
    allvelx = []
    allvely = []
    C_matrix = np.zeros((n_size[-1], snapshot))
    for t in range(time): 
        print('time = %d' %t)

        #updates velocity for every time step then calculates int
        norm_v, pos, posx, posy, velx, vely = update() #applies flocking behaviour and updates velocites and positions for "time" steps. 

        if t > (time_steps - snapshot): #obtain position and velocity data for final snapshots
            allposx.append(posx)
            allposy.append(posy)
            allvelx.append(velx)
            allvely.append(vely)
            #Calculate corelations for a range of neigbourhood sizes. 

            for n in n_size:
                c = calc_c(norm_v, pos, n)
                C_matrix[n-1, time_steps - t] =c

        #calculating C_int by considering only local neigbourhood. 
        C = calc_c(norm_v, pos, 20) #obtaining C int for n = 20 to see if we are in steady state
        C_int.append(C)
    C_avg = np.average(C_int[time_steps - snapshot: time_steps-1])
    C_std = np.std(C_int)
    C_avg_n = np.average(C_matrix, axis=1)
    C_std_n = np.std(C_matrix, axis = 1)

    return allposx, allposy, allvelx, allvely , C_int, C_avg, C_std, C_avg_n, C_std_n, C_matrix


#-----------------------------------------------
#run the simulaiton and post processing code
print('Simulation Info:')
print('n_boids= %d' %boid_n)
#print('nc = %d' %n_size)
print('time_steps = %d' %time_steps)
print('snapshot = %d' %snapshot)
posx, posy, velx, vely, C_int, C_avg, C_std, C_avg_n, C_std_n, C_matrix = run()
print('')
print("C_int = ")
for c in C_int:
    print(c)
print('')
print("C_avg = %f " %C_avg)
print('')
print("C_std = %f " %C_std)
print('')
print("Posistion Data x  = ")
print(posx)
print('')
print("Posistion Data y  = ")
print(posy)
print('')
print("velocity data x =  ")
step = snapshot
print(velx)
print('')
print("velocity data y =  ")
print(vely)
print('')
print("C_avg_n = ")
print(C_avg_n)
print('')
print('C_std_n = ')
print(C_std_n)
print('')
print('C_Matrix = ')
print(C_matrix)
print('')


total_time = time()-start
print('Time taken to run: %d s' %total_time)


