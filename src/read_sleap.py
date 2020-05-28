import h5py

with h5py.File('../data/sleap_0_Diffgroup1-1.h5', 'r') as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]

print(occupancy_matrix)
print(tracks_matrix.shape)
print(tracks_matrix[:,:,:,0])