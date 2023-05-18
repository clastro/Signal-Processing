from scipy import signal
#Short Time Fourier Transform

x = dataloader_obj.__getitem__(0)[0][0] # Example Time Series Data
fs = 256 #Sampling Frequency

f, t, Zxx = signal.stft(x[:,0], fs, nperseg=16) # nperseg : Windows size

n_features = 5
freq_range = int(t.shape[0] / n_features)

# 5 Feature Extraction

abs(Zxx[:,:freq_range]).mean()
abs(Zxx[:,freq_range:2*freq_range]).mean()
abs(Zxx[:,2*freq_range:3*freq_range]).mean()
abs(Zxx[:,3*freq_range:4*freq_range]).mean()
abs(Zxx[:,4*freq_range:5*freq_range]).mean()

"""
#Visualization
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
"""
