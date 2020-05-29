import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#file=pd.read_csv("linear_reg.csv")
file=pd.read_csv("linear_regression.csv")

#extracting input x and output y from the dataset
second_feature=file['z'].values
first_feature=file['y'].values

#since the cost function revolves aroud the matrix and the vertor of the input and output data, changing to matrix
matrix_1=np.mat(first_feature)
output_y=np.mat(second_feature)



#stacking input values with the matrix of one to transpose matrix_1
rows1=np.shape(matrix_1)[1]
ones_matrix=np.ones((1,rows1),dtype=int)
input_X=np.hstack((ones_matrix.T,matrix_1.T))
#hstack stack the data's in horizotal order

# =============================================================================
# we knoe for Locally_weighted_regression the hypothesis is given as,
# h(x)=x*theta
# we have calculated the input x from above stacking. theta can be calculated as,
# theta=(xtrans.weight.x)inverse*(x.transpose*weight*y*transpose)
# =============================================================================


# =============================================================================
# Here we calculate the local_weight for each input x
# ∑𝑖𝑒𝑥𝑝(−|𝑥(𝑖)−𝑥|22𝜏2)(𝑦(𝑖)−𝜃𝑇𝑥(𝑖))
# =============================================================================
def local_weight(point_x,x,t):
    a=np.shape(x)[0]
    weights=np.mat(np.eye(a))
    
    for j in range(a):
        deviation_of_x=point_x-x[j]
        weights[j,j]=np.exp((deviation_of_x*deviation_of_x.T)/(-2.0*t**2))
    return weights
#=============================================================================





# =============================================================================
# Now we implement the equation theta=(xtrans.weight.x)inverse*(x.transpose*weight*y*transpose) to find theta
# 
# =============================================================================

def ind_theta(point_x,x,y,t):
    wt=local_weight(point_x,x,t)
    theta_for_x=(x.T*(wt*x)).I*(x.T*(wt*y.T))
    return theta_for_x
    
# ============================================================================= 

def local_regression(x,y,t):
    x_rows=np.shape(x)[0]
    updated_theta=np.zeros(x_rows)
    
# =============================================================================
#  we have created an enitre matrix  having the same rows as input for calculating the cost function.
#  Now we calculate h(x)=x*theta for each theta

# =============================================================================
    for i in range(x_rows):
        updated_theta[i]=x[i]*ind_theta(x[i],x,y,t)
    return updated_theta


# =============================================================================
    
thau=1.5
hypothesis=local_regression(input_X,output_y,thau)

copy_x=input_X.copy()
copy_x.sort(axis=0)
plt.scatter(first_feature, second_feature, color='red')
plt.plot(copy_x[:, 1], hypothesis[input_X[:, 1].argsort(0)],color="blue",linewidth=3)
plt.xlabel("Fixed acidity")
plt.ylabel("quality")
plt.show()

linr=LinearRegression()
linr.fit(input_X,output_y.reshape(-1,1))

print()
print(hypothesis)

#printing error
mean_absolute=metrics.mean_absolute_error(second_feature,hypothesis)
mean_square=metrics.mean_squared_error(second_feature,hypothesis)
root_mean_error=np.sqrt(mean_square)

print(mean_absolute)
print(mean_square)
print(root_mean_error)