#A1
import math
nums=[115.3,195.5,120.5,110.2,90.4,105.6,110.9,116.3,122.3,125.4]
def Mean(nums):
    size = len(nums)
    sum = 0

    for i in nums:
        sum +=i

    mean=sum/size
    return mean



def Median(nums):
    size = len(nums)

    nums.sort()
    if size % 2 == 0:
        num1=nums[size//2]
          #// is floor divison , return only int value
        num2=nums[size//2 -1]
        num=(num1+num2)/2
    else:
        #if odd num of values =>floor index
        num =(size+1)//2
    return num



def Mode(nums):
    d = dict()

    for i in nums:
        d[i] = 0
    max_rep_num = nums[0]
    for i in nums:
        d[i] +=1
        if d[i] > d[max_rep_num]:
            max_rep_num = i
    if d[max_rep_num] == 1:
        print("Mode is not available")
    else:
        print("The mode is "+str(i)+" : "+str(d[max_rep_num]))


def varience(nums):
    n=len(nums)
    mean=sum(nums)/n
    var=0
    for i  in nums:
      var=var+(i-mean)**2
    vari=var/(n-1)
    return vari


def Standard_deviation(nums):
    n=len(nums)
    mean=sum(nums)/n
    var=0
    for i  in nums:
      var=var+(i-mean)**2
    vari=var/(n-1)
    return math.sqrt(vari)

def min_max_normalization(nums):
    normalized_values=[]
    mini=min(nums)
    maxi=max(nums)
    for i in nums:
        normalized_values.append((i-mini)/(maxi-mini))
    return normalized_values

def standardization(nums):
  standardized_values=[]
  for i in nums:
    standardized_values.append((i-Mean(nums))/Standard_deviation(nums))
  return standardized_values

print("Mean is "+str(Mean(nums)))
print("Median is "+str(Median(nums)))
Mode(nums)
print("Varience is "+str(varience(nums)))
print("SD is "+str(Standard_deviation(nums)));
print("The min-max normalized values are "+str(min_max_normalization(nums)))
print("standardized_data" +str(standardization(nums)))
