import matplotlib.pyplot as plt

plt.plot(C_avg)
plt.xlabel("time")
plt.ylabel("C_exp")
plt.title('n_boids = %d ' %boid_n)
# plt.show()
plt.savefig('SS_check_b%d_s%d ' %boid_n %snapshot)


