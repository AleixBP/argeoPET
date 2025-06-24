from argeoPET import array_lib as np


# From pairs of x,y positions to sinogram parameterization (2D)
def angle_distance_2D(d1, d2, point=None):
    if point is None: point = np.zeros(d1.ndim)
    dif = d2-d1 # thinking of the line as d1+t*dif
    dif /= np.linalg.norm(dif, axis=1)[:, np.newaxis] # normalise
    normal_to_line = (d1-point) - np.einsum('ij,ij->i', d1-point, dif)[:,np.newaxis]*dif # einsum for dot product over all the list #points from point to line
    angles = np.arctan2(*normal_to_line.T[::-1])
    return angles+(angles<0)*np.pi, np.sign(angles)*np.linalg.norm(normal_to_line, axis=1)


# From pairs of x,y,z positions to sinogram parameterization (4D)
def angle_distance_3D(d1, d2, point=None, test=False):
    
    dif = d2-d1
    dif /= np.linalg.norm(dif, axis=1)[:, np.newaxis]
    
    # Compute two angles of vector direction
    #polar = np.arccos(dif[:,0]/1.) # r=1. because normalized #from positive z axis (0) towards plane (pi)
    polar = np.arcsin(dif[:,0]/1.) # from (-pi/2) to (pi/2) # assuming z is axis 0
    azi = np.arctan2(dif[:,1], dif[:,2]) # from -pi to pi # assuming y is axis 1, x is 2
    
    # Compute closest point on line
    if point is None: point = np.zeros(d1.shape[-1])
    normal_to_line = (d1-point) - np.einsum('ij,ij->i', d1-point, dif)[:,np.newaxis]*dif # from origin (point) to closest point on the line
    
    # Get orthogonal basis for each vector following typical spherical coordinates
    # should it not be w2T?
    w2 = np.vstack(
        (-np.sin(azi), np.cos(azi), np.zeros_like(azi))
        ).T
    w3 = np.vstack(
        (-np.cos(azi)*np.sin(polar), -np.sin(azi)*np.sin(polar), np.cos(polar))
        ).T
    
    # Project to get the two components perpendicular to the line vector
    s2 = np.einsum('ij,ij->i', normal_to_line[:,::-1], w2)
    s3 = np.einsum('ij,ij->i', normal_to_line[:,::-1], w3)
    
    if not test:
        # Adjust overparameterization of the angle
        
        return azi + (azi<0)*np.pi, np.sign(azi)*polar, np.sign(azi)*s2, s3
    
    else:
        #https://pi2-docs.readthedocs.io/en/latest/spherical_coordinates.html
        #toft radon page 179
        # Check that the distance to the line is correct
        print(
            np.max(np.abs(
                np.linalg.norm(np.cross(d1, dif), axis=1) 
                - np.linalg.norm(normal_to_line, axis=1)
            ))
        )
        
        # Check orthonormality and others
        w1 = np.vstack(
            (np.cos(azi)*np.cos(polar), np.sin(azi)*np.cos(polar), np.sin(polar))
        ).T
        print( np.einsum('ij,ij->i', w2, w2) ) #1s
        print( np.abs(np.max(np.einsum('ij,ij->i', w2, w3))) ) #0s
        print( np.abs(np.max(np.einsum('ij,ij->i', w1, w2))) ) #0s
        print( np.abs(np.max(np.einsum('ij,ij->i', w1, w3))) ) #0s

        # the next one should be 0s since normal_to_line[:,::-1] is the x,y,z of the point in the line closest
        # to the origin and, therefore, a perpendicular vector to that line
        print( np.max(np.abs(np.einsum('ij,ij->i', normal_to_line[:,::-1], w1) )) )
        
        w1T = np.vstack(
        (np.cos(azi)*np.cos(polar), -np.sin(azi), np.cos(azi)*np.sin(polar))
        ).T
        
        return 0