from hampel import hampel
outlier_indices = hampel(pd.Series(ecg_wave), window_size=5, n=3)
wave_imputation = hampel(pd.Series(ecg_wave), window_size=5, n=3, imputation=True)
pd.Series(ecg_wave).plot(style="k-")
wave_imputation.plot(style="g-")
plt.show()
