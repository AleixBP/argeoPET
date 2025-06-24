from argeoPET import array_lib as np

mouse = True
if not mouse:
    path = "/home/boquet/data/3D_plaque_20min_nobug/"
    file_path = "_Full_20_250Si_0200Kapton_000Bi_2x2_100um_1Chip_3mmCool.conf_different.npy"
    plaque_path = "/home/boquet/data/3D_plaque_20min/Source/Source_Full_20_250Si_0200Kapton_000Bi_2x2_100um_1Chip_3mmCool.conf_different.npy"
elif False:
    path = "/home/boquet/data/voxelmouseplaque_august/"
    file_path = "_Full_20_250Si_0200Kapton_050Bi_2x2_100um_1Chip_3mmCool.conf_different.npy"
    plaque_path = "/home/boquet/data/voxelmouseplaque_august/plaque_2021090101_Full_20_250Si_0200Kapton_050Bi_2x2_100um_1Chip_3mmCool.conf_all.npy"
else:
    path = "/home/saidi/2024_mouse/ini_"
    file_path = "_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"
    path = "/home/saidi/2024_mouse/pla_"
    plaque_path = "_01_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"


def mouse_loader(n_files=40, save=False):
    
    dets = np.load(plaque_path)
    dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
    print(dets.shape)

    for i in range(1, 1+n_files):
        dets1 = np.load(path+str(i).zfill(2)+file_path)
        dets1 = np.array([dets1[:,::2,:].squeeze(), dets1[:,1::2,:].squeeze()])
        dets = np.hstack((dets, dets1))
        print(dets1.shape)
        
    print(dets.shape)
    
    return dets


def all_loader(n_files, path, file_path, save=None):
    
    dets = data_loader(path+str(1).zfill(2)+file_path)
    
    for i in range(2, n_files+1):
            dets1 = data_loader(path+str(i).zfill(2)+file_path)
            dets = np.hstack((dets, dets1))
    
    if save is not None:
        np.save(save, dets)
    
    return dets


def data_loader(file_name, save=False):
    
    dets = np.load(file_name)
    dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
    
    return dets
    
    
    
def print_mateus(rec, file_name):
    data = 255*np.abs(rec)/np.max(rec)#np.finfo(rec.dtype).max
    data = data.astype(np.uint8)

    with open(file_name, "ab") as f:
        for z in range(data.shape[0])[::-1]:
            #f.write(b"\n")
            np.savetxt(f, data[z,:,:], fmt="%u", delimiter='\t', newline='\n')


# Save into files
if False:
    
    if False:
        dets = np.load(plaque_path)
        dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
        np.save("plaque.npy", dets)
    
    else:
        dets = np.load(path+str(1).zfill(2)+file_path)
        dets = np.array([dets[:,::2,:].squeeze(), dets[:,1::2,:].squeeze()])
        
        for i in range(2, 41):
            dets1 = np.load(path+str(i).zfill(2)+file_path)
            dets1 = np.array([dets1[:,::2,:].squeeze(), dets1[:,1::2,:].squeeze()])
            dets = np.hstack((dets, dets1))
            
        np.save("mouse.npy", dets)
    
if False:
    path = "/home/saidi/2024_mouse/ini_"
    file_path = "_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"
    all_loader(7, path, file_path, "mouse_v2.npy")
    
    path = "/home/saidi/2024_mouse/pla_"
    plaque_path = "_01_Full_20_250Si_0300Kapton_050W_2x2_150um_1Chip_3mmCool.conf_all.npy"
    all_loader(2, path, file_path, "plaque_v2.npy")