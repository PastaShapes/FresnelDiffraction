from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

e0 = 8.854e-12
c = 3e8

def Fresnel2dreal(yp, xp, y, x, k, z):
    arg = (k/(2*z)) * ((x-xp)**2 + (y-yp)**2)
    return np.cos(arg)

def Fresnel2dimag(yp, xp, y, x, k, z):
    arg = (k/(2*z)) * ((x-xp)**2 + (y-yp)**2)
    return np.sin(arg)

def Fresnel_intensity(realpart, imagpart):
    return e0 * c * (realpart**2 + imagpart**2)

def Fresnel_error(realpart, realerror, imagpart, imagerror):
    error = 2 * e0 * c * np.sqrt((realpart*realerror)**2 + (imagpart * imagerror)**2)
    return error

def Monte_Carlo_2D(myfunc, xmin, xmax, ymin, ymax, N, args):
    """
    Parameters
        myfunc : Function
            The function to be integrated.
        xmin : float
            Minimum value of range for integration.
        xmax : float
            Maximum value of range for integration.
        ymin : float
            Minimum value of range for integration.
        ymax : float
            Maximum value of range for integration.
        N : int
            Number of MC points.
        args : tuple
            Additional arguments / constants
    Returns
        integral : float
            The value of the integral.
        error : float
            The absolute error in the integral.
    """

    xsamples = np.random.uniform(low=xmin, high=xmax, size=N)
    ysamples = np.random.uniform(low=ymin, high=ymax, size=N)

    values = myfunc(xsamples, ysamples, *args)

    mean = values.sum() / N
    meansq = (values * values).sum() / N

    area_integral = (xmax - xmin) * (ymax - ymin) * mean
    error = (xmax - xmin) * np.sqrt((meansq - mean**2) / N)

    return (area_integral, error)

def get_config():
    print("A. Fresnel (Near-Field)")
    print("B. Fraunhofer (Far-Field)")
    
    choice = input("Enter A or B: ").strip().lower()
    
    if choice == 'a':
        return {
            'z': 0.005,
            'width': 2e-4, 
            'screen_limit': 0.0005,
        }
    else:
        return {
            'z': 0.05,
            'width': 2e-5,         
            'screen_limit': 0.005,          
        }

def part_1(config):
    aperture_width = config['width']

    xp1 = aperture_width / 2
    xp2 = -xp1
    yp1 = xp1
    yp2 = -xp1
    k = 2*np.pi / 300e-9
    x = 0
    y = 0
    z = config['z']

    numpoints = 200
    limit = config['screen_limit']
    xvals = np.linspace(-limit, limit, num=numpoints)
    yvals = np.zeros(numpoints)
    yerrors = np.zeros(numpoints)

    for i in range(numpoints):
        x = xvals[i]
        realpart, realerror = integrate.dblquad(Fresnel2dreal, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        imagpart, imagerror = integrate.dblquad(Fresnel2dimag, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        value = Fresnel_intensity(realpart, imagpart)
        yvals[i] = value
        value_error = Fresnel_error(realpart, realerror, imagpart, imagerror)
        yerrors[i] = value_error

    plt.figure(figsize=(8, 5))
    plt.grid(True)
    plt.xlabel("Screen coordinate (m)")
    plt.ylabel("Relative intensity")
    plt.errorbar(xvals, yvals, yerr=yerrors, fmt="-", capsize = 3, ecolor="gray", label="Intensity")
    plt.show()

def part_2(config):
    aperture_width = config['width'] # size of aperture in both directions (m)
    xp1 = -aperture_width / 2
    xp2 = aperture_width / 2
    yp1 = -aperture_width / 2
    yp2 = aperture_width / 2

    screen_distance = config['z'] # distance from aperture to screen (m)
    wavelength = 589e-9 # wavelength of light used (m)

    x1 = config['screen_limit'] #0.01   # lower x-limit for screen (m)
    x2 = -x1     # upper x-limit for screen (m)
    y1 = x1      # lower y-limit for screen (m)
    y2 = x2      # upper x-limit for screen (m)
    numpoints = 100 # number of sample points in each direction

    k = 2*np.pi / wavelength

    xvals = np.linspace(x1, x2, numpoints)
    yvals = np.linspace(x1, x2, numpoints)
    intensity = np.zeros((numpoints, numpoints))

    for i in range(numpoints):
        for j in range(numpoints):
            realpart, realerror = integrate.dblquad(Fresnel2dreal, xp1, xp2, yp1, yp2, args=(yvals[i], xvals[j], k, screen_distance))
            imagpart, imagerror = integrate.dblquad(Fresnel2dimag, xp1, xp2, yp1, yp2, args=(yvals[i], xvals[j], k, screen_distance))
            intensity[i, j] = Fresnel_intensity(realpart, imagpart)

    extents = (x1, x2, y1, y2) # Sets the limits for the plot
    plt.imshow(intensity,vmin=0.0,vmax=1.0*intensity.max(),extent=extents,
            origin="lower",cmap="nipy_spectral_r")    # Try different
    #           origin="lower",cmap="nipy_spectral")  # colour
    #           origin="lower",cmap="gist_stern")     # maps
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title('Rectangular diffraction;\nz = {:4.2f}m'.format(screen_distance))
    plt.colorbar()
    plt.show()

def part_3(config):
    aperture_width = config['width'] # size of aperture in both directions (m)
    xp1 = -aperture_width / 2 
    xp2 = aperture_width / 2
    
    def yp1func(xp): return -np.sqrt((aperture_width / 2)**2 - xp**2)
    def yp2func(xp): return np.sqrt((aperture_width / 2)**2 - xp**2)

    screen_distance = config['z'] # distance from aperture to screen (m)
    wavelength = 589e-9 # wavelength of light used (m)

    x1 = config['screen_limit'] # lower x-limit for screen (m)
    x2 = -x1     # upper x-limit for screen (m)
    y1 = x1      # lower y-limit for screen (m)
    y2 = x2      # upper x-limit for screen (m)
    numpoints = 100 # number of sample points in each direction

    k = 2*np.pi / wavelength

    xvals = np.linspace(x1, x2, numpoints)
    yvals = np.linspace(x1, x2, numpoints)
    intensity = np.zeros((numpoints, numpoints))

    for i in range(numpoints):
        for j in range(numpoints):
            realpart, realerror = integrate.dblquad(Fresnel2dreal, xp1, xp2, yp1func, yp2func, args=(yvals[i], xvals[j], k, screen_distance), epsabs=1e-3, epsrel=1e-3)
            imagpart, imagerror = integrate.dblquad(Fresnel2dimag, xp1, xp2, yp1func, yp2func, args=(yvals[i], xvals[j], k, screen_distance), epsabs=1e-3, epsrel=1e-3)
            intensity[i, j] = Fresnel_intensity(realpart, imagpart)

    extents = (x1, x2, y1, y2) # Sets the limits for the plot
    plt.imshow(intensity,vmin=0.0,vmax=1.0*intensity.max(),extent=extents, origin="lower",cmap="nipy_spectral_r")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title('Circular diffraction;\nz = {:4.2f}m'.format(screen_distance))
    plt.colorbar()
    plt.show()

def part_4(config):
    aperture_width = config['width'] # size of aperture in both directions (m)
    xp1 = -aperture_width / 2
    xp2 = aperture_width / 2
    yp1 = -aperture_width / 2
    yp2 = aperture_width / 2

    z = config['z'] # distance from aperture to screen (m)
    
    wavelength = 589e-9 # wavelength of light used (m)

    x1 = config['screen_limit'] # lower x-limit for screen (m)
    x2 = -x1       # upper x-limit for screen (m)
    y1 = x1        # lower y-limit for screen (m)
    y2 = x2        # upper x-limit for screen (m)
    numpoints = 200 # number of sample points in each direction

    k = 2*np.pi / wavelength

    N = 10000 # number of MC samples

    xvals = np.linspace(x1, x2, numpoints)
    yvals = np.linspace(x1, x2, numpoints)
    intensity = np.zeros((numpoints, numpoints))

    def Fresnel2dreal_masked(yp, xp, y, x, k, z):
        arg = (k/(2*z)) * ((x-xp)**2 + (y-yp)**2)
        within_margin = (xp**2 + yp**2) <= (aperture_width / 2)**2
        return np.cos(arg) * within_margin

    def Fresnel2dimag_masked(yp, xp, y, x, k, z):
        arg = (k/(2*z)) * ((x-xp)**2 + (y-yp)**2)
        within_margin = (xp**2 + yp**2) <= (aperture_width / 2)**2
        return np.sin(arg) * within_margin

    for i in range(numpoints):
        for j in range(numpoints):
            realpart, realerror = Monte_Carlo_2D(Fresnel2dreal_masked, xp1, xp2, yp1, yp2, N, args = (yvals[i], xvals[j], k, z))
            imagpart, imagerror = Monte_Carlo_2D(Fresnel2dimag_masked, xp1, xp2, yp1, yp2, N, args = (yvals[i], xvals[j], k, z))
            intensity[i, j] = Fresnel_intensity(realpart, imagpart)

    extents = (x1, x2, y1, y2) # Sets the limits for the plot
    plt.imshow(intensity,vmin=0.0,vmax=1.0*intensity.max(),extent=extents, origin="lower",cmap="nipy_spectral_r")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title('Circular diffraction;\nz = {:4.2f}m'.format(z))
    plt.colorbar()
    plt.show()

my_input = ""
while my_input.lower().strip() != "q":
    print("Fresnel diffraction simulator")
    print("1 - Line Scan (1D)")
    print("2 - Rectangular Aperture (Scipy)")
    print("3 - Circular Aperture (Scipy)")
    print("4 - Circular Aperture (Monte Carlo)")
    print("Q - Quit")

    my_input = input("\nPlease enter '1', '2', '3', '4', or 'q': ").strip().lower()

    if my_input in ['1', '2', '3', '4']:
        current_config = get_config()
        
        if my_input == '1':
            part_1(current_config)
        elif my_input == '2':
            part_2(current_config)
        elif my_input == '3':
            part_3(current_config)
        elif my_input == '4':
            part_4(current_config)
            
    elif my_input != 'q':
        print("Invalid choice, please try again.")