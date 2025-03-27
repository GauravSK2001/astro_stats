import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import tqdm 



## Importing the data

d1=np.loadtxt('./P1data-2024/P1data01.txt')
d2=np.loadtxt('./P1data-2024/P1data02.txt')
d3=np.loadtxt('./P1data-2024/P1data03.txt')
d4=np.loadtxt('./P1data-2024/P1data04.txt')
d5=np.loadtxt('./P1data-2024/P1data05.txt')
d6=np.loadtxt('./P1data-2024/P1data06.txt')

data=[d1,d2,d3,d4,d5,d6] # List of all the data

## Plotting the data

fig , ax = plt.subplots(2,3,figsize=(15,10))

for i in range(6):
    ax[i % 2, i // 2].scatter(data[i][:, 0], data[i][:, 1], marker='.', s=0.5, alpha=1)
    ax[i % 2, i // 2].set_title(f'Data Set {i + 1}',fontsize=15)
    ax[i % 2, i // 2].set_ylim(0,1000)
    ax[i % 2, i // 2].set_xlim(0,1000)
    ax[i % 2, i // 2].set_xlabel('X',fontsize=12)
    ax[i % 2, i // 2].set_ylabel('Y',fontsize=12)
plt.tight_layout()
plt.savefig('./Data/Dataset.pdf')
plt.show()

## Function for RR, DD, DR

def count_nei(data,bins,Length_of_random=20,Cross=False): # to count the number of neighbours
    bins=np.linspace(1000/bins,1000,bins) # First bin is the smallest distance
    Data_tree = sc.spatial.KDTree(data) # Creating a KDTree for the data

    R= [np.random.uniform(0,1000,(len(data),2)) for _ in (range(Length_of_random))] # Creating random data
    Random_tree = [sc.spatial.KDTree(r) for r in R] # Creating a KDTree for the random data
    
    DD = Data_tree.count_neighbors(Data_tree, bins,cumulative=False) # Counting the number of neighbours for the data
    RR = np.mean([r.count_neighbors(r, bins,cumulative=False) for r in Random_tree],axis=0) # Counting the number of neighbours for the random data
    if Cross: # If we want to calculate DR as well
        DR = np.mean([Data_tree.count_neighbors(r, bins,cumulative=False) for r in Random_tree],axis=0) 
        return DD,RR,DR # Returning the DD, RR, DR if Cross is True
    
    return DD,RR # Returning the DD, RR if Cross is False

def Two_point_normal(data,bins,Length_of_random=20): # Natural estimator

    DD,RR=count_nei(data,bins,Length_of_random) 

    return DD/RR - 1 # w_1

def Two_point_Landy(data,bins,Length_of_random=20): # Landy-Szalay estimator
    
    DD,RR,DR=count_nei(data,bins,Length_of_random,Cross=True)
    
    DD[0]-=len(data) # to remove the self count
    RR[0]-=len(data) # to remove the self count
    DD=DD/2.0 # to remove the double count
    RR=RR/2.0 # to remove the double count

    return (DD/RR) - (1-(1/len(data)))*(DR/RR) +1 # w_4

def plotter(data,bins,Length_of_random=20): # to plot the two point correlation function
 
    fig, ax = plt.subplots(3,2,figsize=(15, 15))#,sharex=True,sharey=True)
    bin=np.linspace(0,1000,bins)
    for i in tqdm.tqdm(range(len(data))):
        w1=Two_point_normal(data[i],bins,Length_of_random)
        w2=Two_point_Landy(data[i],bins,Length_of_random)
        ax[i % 3, i // 3].step(bin[0:],w1[0:],'r',label='Normal Estimator',where='mid')
        ax[i % 3, i // 3].step(bin[0:],w2[0:],'b',label='Landy-Szalay Estimator',where='mid')
        if i==0:
            ax[i % 3, i // 3].legend(loc='lower left',fontsize=12,shadow=True, fancybox=True,frameon=True)
        ax[i % 3, i // 3].set_title(f'Data Set {i + 1}',fontsize=15)
        # ax[i % 3, i // 3].set_ylim(0,1000)
        ax[i % 3, i // 3].set_xlim(0,1000)
        ax[i % 3, i // 3].set_xlabel(r'Separation $(\theta)$',fontsize=12)
        ax[i % 3, i // 3].set_ylabel(r'Estimator $(w(\theta))$',fontsize=12)
        ax[i % 3, i // 3].xaxis.set_minor_locator(plt.MultipleLocator(25))
        ax[i % 3, i // 3].xaxis.set_major_locator(plt.MultipleLocator(100))
        ax[i % 3, i // 3].yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax[i % 3, i // 3].yaxis.set_minor_locator(plt.MultipleLocator(0.0025))
        ax[i % 3, i // 3].grid(True, which='major', linestyle='-', linewidth=0.5)
        ax[i % 3, i // 3].grid(True, which='minor', linestyle='--', linewidth=0.5,alpha=0.2)
    plt.tight_layout()
    plt.savefig('./Data/Estimators.pdf')
    plt.show()
plotter(data,100,20) # Plotting the two point correlation function with 100 bins and 20 random data