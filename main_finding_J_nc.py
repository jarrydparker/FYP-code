import numpy as np
from time import time
import matplotlib.pyplot as plt

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
boid_n = 50 #number of boids
snapshot = 5 #how many snapshots do we use to calculate the interaction parameters
time_steps = 50
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
# for loop to run through n_c values
n_c_list = list(range(5,50))
J_n_c = []
prob = []
log_likelihood = []
# for n in n_c_list
def run(time, n_c): 
    #function to run simulation code
    #time = how many timesteps do we want to simulate for
    C_int = []
    C_J_int = []
    J = np.linspace(0,10,101)
    J_const=1
      
    for t in range(time): 
        if n_c > boid_n:
            print("ERROR: nc >n_boids")
            break 
        #updates velocity for every time step then calculates int
        norm_v, pos = update() #applies flocking behaviour and updates velocites and positions for "time" steps. 

        #calculating C_int by considering only local neigbourhood + expected J values. 
        list_of_sums = []
        n_ij_list_of_sums = []
        for i in range(boid_n):
            #find n_c closest boids
            idx_list = find_nearest(pos[i],pos,n_c)
            list_of_products = []
            n_ij = []
            for j in idx_list:
                n_idx_list = find_nearest(pos[j],pos,n_c)
                if i in n_idx_list: # is an element of n_idx_list
                    n_ij.append(1)
                else:
                    n_ij.append(1/2)
                a = norm_v[i]
                b = norm_v[j]
                product = np.inner(a, b)
                list_of_products.append(product)
            ## Finding weighted list of dot products due to being in the neighbourhood    
            product_sum = np.sum(list_of_products)/n_c
            weighted_lop = np.dot(n_ij,list_of_products)
            n_ij_product_sum = np.sum(weighted_lop)/n_c # /n_c #may need to add /n_c
            n_ij_sum = np.sum(n_ij)
            
            list_of_sums.append(product_sum)
            n_ij_list_of_sums.append(n_ij_product_sum)
            # Making J array to plot graph
            J_list_of_sums = n_ij_list_of_sums*J_const
            # print('list of sums:',list_of_sums)   
            
        # print('time = %d' %t)
        C = np.sum(list_of_sums)/boid_n
        C_int.append(C)
        C_J1 = np.sum(J_list_of_sums)/boid_n
        C_J = J*np.sum(J_list_of_sums)/boid_n
        C_J_int.append(C_J1)        
        J_estim = np.divide(C_int,C_J_int)
        # J_estim_list = np.append(J_estim)

    #-------------------Finding C_int in steady state-----------------#
    # plotting average correlations over timestep - check we are in steady state.
    # Obtain the experimental value of C_int
    # printing experimental value of average correlation
    # check if simulation was in steady state
    #get avg for C_J and C_int to compare J values
    C_avg = np.average(C_int[(t-snapshot):(t-1)])
    # C_avg_list = np.average(C_avg)
    C_J_avg = np.average(C_J_int[(t-snapshot):(t-1)])
    # C_J_avg_list = np.append(C_J_avg) 
    
    # Will need list of J vals to find probability
    J_expected = 1/(n_c/2*(1-C_avg))
    # J_expected_list = np.append(J_expected)
    J_exp = np.divide(C_avg,C_J_avg)
    # J_exp_list = np.append(J_exp)
    print('J_expected = ',J_expected)
    print('J Estimate: ',np.average(J_estim))
    print('J_Exp: ',J_exp)
    print('C_avg = ', C_avg)
    print('C_J_avg: ',C_J_avg)
    # print('n_boids = %d' %boid_n)
    # print('nc  = %d' %n_c )
    # print('time_steps = %d' %time_steps)
    # print('snapshot = %d' %snapshot)
    # print('')
    # print('')
    # print('C_int Results:')
    # print(C_int)
    # print('C_J_int: ',C_J_int)
    # print(C_J_int)

#-------------- J vs C_avg Plot ----------------
    # plt.figure()
    # plt.plot(J,C_J)
    # plt.axhline(C_avg, linestyle = '-')
    # plt.title('J vs C_int') 
    # plt.xlabel('J')
    # plt.ylabel('C_int')
    lnZ = J_exp*n_ij_sum/2
    l_likelihood = (-lnZ+J_exp*boid_n*n_c*C_avg/2)/boid_n
    log_likelihood.append(l_likelihood)
    print('Log Likelihood:' ,l_likelihood)
    return l_likelihood
#----------JARRYD ANIMATION CODE--------------
#make this section easy to comment out. 


#----------------------------------------------

#--------------JARRYD ANALYSIS CODDE-----------
#Find J and nc 
#Split J into J*nij
# nij = 1 if j is an element of n_c for the ith bird and vice versa
# nij = 1/2 if j is an element of n_c for the ith bird but i is not an element for the jth and vice versa
# o elsewhere
# this allows J to be purely the interaction strength
# not sure if the averages will work as the nij matrix depends on boids in that neighbourhood
# for boid in flock
# a = update()
# print(a[1][0])
# print(find_nearest(a[1][0],a[1],n_size))

#-----------------------------------------------

#----------------NIKKI ANALYSIS CODE-----------
#Find relationship between J and nc 
#Confirm the max entropy model agrees with simulation by plotting
#the correlation function from the max entropy model vs the one from simulation

#correlation as a function of ditance 
#perpendicular component
#longitudinal component
#average is performed over all pairs seperated by distance r
#corealation as a function of distance





#-----------------------------------------------
#run the simulaiton and post processing code
log_likelihood2 = []
for n in n_c_list:
    print(n)
    ll = run(time_steps,n)
    log_likelihood2.append(ll)

plt.plot(n_c_list,log_likelihood2)

total_time = time()-start
print('Time taken to run: %d s' %total_time)
