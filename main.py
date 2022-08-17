import numpy as np
import matplotlib.pyplot as plt

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

width = 1000
height = 1000

flock = [Boid(np.random.rand()*1000, np.random.rand()*1000, width, height) for _ in range(10)]

def draw():
    global flock

    vel = []
    for boid in flock:
        boid.edges()
        boid.apply_behaviour(flock)
        boid.update()
        vel.append(boid.velocity)
    #average velocity for timestep
    avg_vel = np.average(np.linalg.norm(vel))
    
    return avg_vel

def run(time = 2000): 
    avg_velocity = []
    for t in range(time): 
        avg_vel = draw()
        avg_velocity.append(avg_vel)


    print("done")
    print(avg_velocity)
    #printing avg velocities
    plt.plot(avg_velocity)
    plt.xlabel("time")
    plt.ylabel("average velocities")
    plt.show()

run()

# TO DO -------------------------------------------------
#(1) each bird is assigned a state label v_i - DONE

#(2) also define a normalised velocity 

#(3) Assume birds are in a statistically stationary state
#(check). - DONE? look at plots to check we are in stationary state
#trial and error, see aprox how long it takes to get stationary state. run code based on that
#after analysis look at plot to confirm system was in stationaty state

#(4) Calculate C_int (eq. 14) for a single flock at a 
#given instant of time

#(5) Compare C_int to experimental value C_exp to fix the 
#experimental value of J for the snapshot

#(6) Plot the log liklihood as a function of n_c. choose n_c
#such that the log liklihood is maximised 

#(7) repeat the procedure for multiple snapshots in the flock
#find the mean and standard deviation of the interaction parameters

#(8) Calculate correlations as a function of distance by fixing J and n_c

#(9) Confirm the max entropy model agrees with simulation by plotting
#the correlation function from the max entropy model vs the one 
#obtained from simulation

