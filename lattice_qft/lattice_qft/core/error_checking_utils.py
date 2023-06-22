import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    MILD = '\033[1;43m'

def typeerror_check(type, *params):
    if any (not isinstance(param, type)  for param in params):
            raise TypeError("{Type} expected for {params} parameters".format(params=params, Type=type))

def real_positive_check(*params):
      typeerror_check((int,float), *params)
      if any (not param > 0 for param in params):
            raise ValueError("{params} is/are not positive real value(s)".format(params=params))

class VersionError(Exception):
      def __init__(self, pkg_name, version_value, current_version=None):
            if current_version is not None:
                  print(bcolors.WARNING + '{pkg} may not operate as intended because version {version} of {pkg} is expected. Please install {pkg} {version} instead of {current} which may is depreciated/appreciated.'.format(pkg=pkg_name, version=version_value, current=current_version) + bcolors.ENDC)
            if current_version is None:
                  print(bcolors.WARNING + '{pkg} may not operate as intended because version {version} of {pkg} is expected. Please install {pkg} {version}.'.format(pkg=pkg_name, version=version_value) + bcolors.ENDC)            

class null:
    pass
class DimensionError(Exception, null):
      
    def __init__(self, array, shape):
          self.shape = [shape]
          self.array_shape = [np.shape(array)]
          temp = []
          if null in self.shape[0]:
                for i in [*shape]:
                      if i is not null:
                        temp.append(i)
                #self.shape = tuple(temp)
                #print(list(self.array_shape[0])[:len(temp)])
                if temp != list(self.array_shape[0])[:len(temp)]:
                    print("{array_s} not in expected shape {shape}".format(array_s=self.array_shape, shape=shape))
                
                      
                
                
          if self.array_shape != self.shape:
                print("{array_s} not in expected shape {shape}".format(array_s=self.array_shape, shape=shape))

class DivisionError(Exception):
     def __init__(self, errormsg: object):
          super().__init__(errormsg)

#raise DivisionError(4)
# DimensionError([[[1]]], (3, 1, null))



            

