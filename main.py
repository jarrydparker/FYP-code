import numpy as np
import matplotlib.pyplot as plt
from time import time 

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

width = 1500
height = 1500
boid_n = 500 #number of boids
snapshot = 250
time_steps = 2000

flock = [Boid(np.random.rand()*1000, np.random.rand()*1000, width, height) for _ in range(boid_n)]

def update():
    global flock

    vel = []
    for boid in flock:
        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        vel.append(boid.velocity)

    #Return the velocities for each time step   
    return vel



def run(time = time_steps): 
    #function to run simulation code
    #time = how many timesteps do we want to simulate for
    correlations = []
    C_avg = [] #average correlation for each time step (see when this becomes steady state) 

    for t in range(time): 
        vel = update() #applies flocking behaviour and updates velocites and positions for "time" steps. 
        norm_v = []
        for v in vel:
            norm_v.append(v/np.linalg.norm(v)) #getting normalised velocites

        #calculating correlation matrix for each time step
        #TO DO: make this part of the code more efficient by using math operations
        #insead of for loops. 
        C_matrix = np.zeros((boid_n, boid_n))
        for i in range(boid_n):
            for j in range(boid_n):
                a = norm_v[i]
                b = norm_v[j]
                C_matrix[i,j] = np.inner(a, b)
                #how alligned are the boids?
        
        #calculating C_int 
        C_int = np.average(C_matrix)
        #saving correlation matric and C_int for each time step. 
        correlations.append(C_matrix)
        C_avg.append(C_int)

    #-------------------Finding C_int in steady state-----------------#
    print("done")
    #plotting average correlations over timestep - check we are in steady state.
    #Obtain the experimental value of C_int
    C_exp = np.average(C_avg[(t-snapshot):(t-1)])
    #printing experimental value of average correlation
    print("C_exp = ", C_exp)
    #check if simulation was in steady state
    plt.plot(C_avg)
    plt.xlabel("time")
    plt.ylabel("C_exp")
    plt.title('n_boids = %d ' %boid_n)
   # plt.show()
    plt.savefig('SS_check_b%d_s%d ' %boid_n %snapshot)

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



#-----------------------------------------------

#run the simulaiton and post processing code
run()
print('Time taken to rin: {time() - start} s')
