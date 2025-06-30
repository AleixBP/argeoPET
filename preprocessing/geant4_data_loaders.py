from argeoPET import array_lib as np
# Load files from Geant .npy outputs into a shape of (2, number_of_LORS, 3), where the 3 stands for (x,y,z)

def mouse_loader(n_files, path, file_path, plaque_path, save=False, zfill=2):
    
    dets = np.load(plaque_path)
    dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
    print(dets.shape)

    for i in range(1, 1+n_files):
        dets1 = np.load(path+str(i).zfill(zfill)+file_path)
        dets1 = np.array([dets1[:,::2,:].squeeze(), dets1[:,1::2,:].squeeze()])
        dets = np.hstack((dets, dets1))
        print(dets1.shape)
        
    print(dets.shape)
    
    return dets


def all_loader(n_files, path, file_path, save=None, zfill=2):
    
    dets = data_loader(path+str(1).zfill(zfill)+file_path)
    
    for i in range(2, n_files+1):
            dets1 = data_loader(path+str(i).zfill(zfill)+file_path)
            dets = np.hstack((dets, dets1))
    
    if save is not None:
        np.save(save, dets)
    
    return dets


def data_loader(file_name, save=False):
    
    dets = np.load(file_name)
    dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
    
    if save is not None:
        np.save(save, dets)
    
    return dets
    
    
def print_mateus(rec, file_name): # Transform end reconstruction into "mateus" format for 3D viz
    data = 255*np.abs(rec)/np.max(rec)#np.finfo(rec.dtype).max
    data = data.astype(np.uint8)

    with open(file_name, "ab") as f:
        for z in range(data.shape[0])[::-1]:
            #f.write(b"\n")
            np.savetxt(f, data[z,:,:], fmt="%u", delimiter='\t', newline='\n')


if False:
    path = "/home/saidi/2024_mouse/ini_"
    file_path = "_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"
    #all_loader(7, path, file_path, "mouse_background.npy", zfill=2)
    
    path = "/home/saidi/2024_mouse/pla_"
    plaque_path = "03_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"
    #data_loader(path+file_path, "plaque.npy")