import numpy as np
from time import time 

#Bialek paper used 1246 boids
#try nc  = 20 birds

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
width = 1500
height = 1500
boid_n = 500 #number of boids
snapshot = 250 #how many snapshots do we use to calculate the interaction parameters
time_steps = 2000
n_size = 20 #neigbourhood size

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
    for boid in flock:
        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        norm_vel.append(boid.velocity/np.linalg.norm(boid.velocity))
        pos.append(boid.position)

    #Return the velocities for each time step   
    return norm_vel, pos


#////////////////////RUN SIMULATION////////////////////////

def run(time = time_steps, n_c = n_size): 
    #function to run simulation code
    #time = how many timesteps do we want to simulate for
    C_int = []

    for t in range(time): 
        if n_c > boid_n:
            print("ERROR: nc >n_boids")
            break 
        #updates velocity for every time step then calculates int
        norm_v, pos = update() #applies flocking behaviour and updates velocites and positions for "time" steps. 

        #calculating C_int by considering only local neigbourhood. 
        list_of_sums = []

        for i in range(boid_n):
            #find n_c closest boids
            idx_list = find_nearest(pos[i],pos,n_c)
            list_of_products = []
            for j in idx_list:
                a = norm_v[i]
                b = norm_v[j]
                product = np.inner(a, b)
                list_of_products.append(product)
            product_sum = np.sum(list_of_products)/n_c
            list_of_sums.append(product_sum)
            
        print('time = %d' %t)
        C = np.sum(list_of_sums)/boid_n
        C_int.append(C)

    #-------------------Finding C_int in steady state-----------------#
    #plotting average correlations over timestep - check we are in steady state.
    #Obtain the experimental value of C_int
    #printing experimental value of average correlation
    #check if simulation was in steady state
    C_avg = np.average(C_int[(t-snapshot):(t-1)])
    print('C_avg = %d ' %C_avg)
    print('n_boids = %d' %boid_n)
    print('nc  = %d' %n_c )
    print('time_steps = %d' %time_steps)
    print('snapshot = %d' %snapshot)
    print('')
    print('')
    print('C_int Results:')
    print(C_int)


#----------JARRYD ANIMATION CODE--------------



#----------------------------------------------

#--------------JARRYD ANALYSIS CODDE-----------
#Find J and nc 

#-----------------------------------------------

#----------------NIKKI ANALYSIS CODE-----------
#Find relationship between J and nc 
#Confirm the max entropy model agrees with simulation by plotting
#the correlation function from the max entropy model vs the one from simulation
#(plotted correlation as a function of different parameters - refer to paper)

#correlation as a function of ditance 
#perpendicular component
#longitudinal component
#average is performed over all pairs seperated by distance r
#corealation as a function of distance


#-----------------------------------------------
#run the simulaiton and post processing code
run()
total_time = time()-start
print('Time taken to run: %d s' %total_time)
