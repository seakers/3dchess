import numpy as np

def main():
    """ 
    Approximates the look angle as a function of the angle 
    between a satellite's and a ground point's position vector

    Used for sanity checks
    """
    Re = 6371.0                                 # radius of the Earth [km]
    b = 7078.0                            
    a = Re

    assert b >= a

    c_max = np.sqrt( b**2 - a**2 )
    gamma_max = np.arccos((a**2 + b**2 - c_max**2) / (2*a*b)) * 180 / np.pi
    print(gamma_max)
    
    n = 100
    dgamma = (gamma_max + 1) / float(n)
    gammas : list = [i*dgamma for i in range(n)]    # angle between GP and SAT position vectors

    look_angles : list[float] = []
    for gamma in gammas:
        c = np.sqrt( a**2 + b**2 - 2*a*b*np.cos(gamma * np.pi/180) )
        alpha = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180 / np.pi
        beta =  np.arccos((a**2 + c**2 - b**2) / (2*a*c)) * 180 / np.pi
        
        look_angle = alpha if beta >= 90.0 else np.NAN
        look_angles.append(look_angle)

        print(gamma, look_angle)
    

if __name__ == '__main__':
    main()