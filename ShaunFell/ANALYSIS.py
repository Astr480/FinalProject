import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import itertools
from itertools import product
import os
from scipy import optimize
from scipy.stats import iqr
qso_file = "Macleod_qso_data/qso.dat"


""" NOTES
		### DONE ###  Make redshift versus mag
		### DONE ###  Make redshift versus variability
		### DONE ###  Make color versus variability
		### DONE ###  Make filter-mag versus number of QSOs
		### DONE ###  Make Variability versus Luminosity
"""


class QSO_analysis():
	"""
	This class holds all the methods for my data analysis. 
	"""
	sec_to_year = (60*60*24*365)**(-1)
	bin_width = 1/12 #years
	
	def __init__(self, filename=qso_file, bin_width = bin_width, makeplots = False):
		"""
		Set various parameters common to all class methods. This includes save directory, plotting styles, file extensions, etc.
		After that, it retrieves the data from the qso.dat file
		"""
		self.file_obj = open(filename, 'rb')
		self.lines = self.file_obj.readlines()
		self.bin_width = bin_width
		self.plot = makeplots
		self.plotstyle = plt.style.use('fivethirtyeight')
		self.savedir = os.path.dirname(os.path.realpath(__file__)) + '/Project Paper/Figures/'
		self.save = True
		self.extension = ".png"
		self.dpi = 900
		self.plot_scale = ["linear", "linear"] #[x,y] plot scale
		self.filters = ['u', 'g', 'r', 'i', 'z']
		self.wavelength = [354.3, 477., 623.1, 762.5, 913.4] #u 3543 g 4770 r 6231 i 7625 z 9134
		self.colors = ['violet', 'green', 'red', 'brown', 'black']
		self.markers = ['<', '>', '^', '*', 'o']
		self.linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(3,1,1,1,1,1))]
		self.linewidth = 1
		self.retrieve_data()
	
	def full_analysis(self, **kwargs):
		"""
		Run the entire analysis. This runs all the class methods that contribute to the analysis such as structure function calculation, different correlations, number density plots, etc.
		"""
		self.retrieve_data()
		
		#set various parameters of analysis
		for key,arg in kwargs.items():
			setattr(self, key, arg)
		
		print("Making structure functions")
		#make structure function for each filter
		for fil in self.filters:
			self.plot_structure_function(filtr = fil)
		
		print("Making luminosity distribution")
		#make luminosity distribution
		self.mag_v_N()
		
		print("Making wavelength versus variability")
		#make color versus variability
		self.color_v_variability()
		
		print("Making redshift versus magnitude")
		#make redshift versus magnitude
		for fil in self.filters:
			self.redshift_v_mag(filter=fil)
		
		print("Making variability versus redshift")
		#make variability versus redshift
		for fil in self.filters:
			self.variability_v_redshift(filtr = fil)
		
		print("Making luminosity versus variability")
		#make luminosity versus variability
		for fil in self.filters:
			self.variability_v_luminosity(filtr = fil)
		
		print("Making quasar count versus redshift")
		self.z_v_N()
		
		print("Making second structure function")
		for fil in self.filters:
			self.plot_structure_function_two(filtr=fil)
			
		
		
	def retrieve_data(self):
		"""
		Retrieves the data from the qso.dat file. 
		
		
		"""
		convert_to_list = lambda bit_obj: list(filter(None, bit_obj.decode().rstrip('\n').split(' ')))#convert bit to string, split by space delimiter, filter out empty strings (spaces), convert to list, take all table names
		self.col_names = convert_to_list(self.lines[6])[1:]
		for i in range(21, len(self.col_names)):
			self.col_names[i]+='_second'
		self.data_list = np.asarray([list(map(float,convert_to_list(self.lines[i]))) for i in range(8, len(self.lines))]) #convert each lines to list, then change all elements to float. Data is list of lists
				
		iter = 0
		for key in self.col_names:
			setattr(self, key, self.data_list[:,iter])
			iter+=1
		self.MJD_R = Time(self.MJD_R, format='mjd')
		self.MJD_R_second = Time(self.MJD_R_second, format='mjd')
		
	def make_structure_function(self, filter = 'u', data = None, bins = []):
		"""
		Calculates the structure function for the filter passed as a parameter.
		"""
		if data==None:
			mag1, mag2 = self.get_filter_mag(filter)
		else:
			mag1, mag2 = data[0], data[1]
		
		self.delta_mags, self.no_outlier_indices = self.remove_outliers(np.abs(mag2-mag1))
		if len(bins)==0:
			bins = np.arange(len(self.u_PSF))
		redshift_correction = np.asarray(self.redshift[bins][self.no_outlier_indices] + 1)
		self.time_lag = np.asarray((np.abs((self.MJD_R_second[bins] - self.MJD_R[bins]).sec)*self.sec_to_year)[self.no_outlier_indices])/redshift_correction
		self.bins = self.make_bins(self.time_lag, 1/12) #bins spacing is 1/12 of a year
		self.digitizing_bins = np.digitize(self.time_lag, self.bins) #returns the indices that bin the time_lag data
		self.digitized_time_lag = np.asarray([np.mean(self.time_lag[self.digitizing_bins==i]) for i in range(len(self.digitizing_bins)) if len(self.time_lag[self.digitizing_bins==i])!=0])
		self.digitized_delta_mags = np.asarray([self.delta_mags[self.digitizing_bins==i] for i in range(len(self.digitizing_bins)) if len(self.delta_mags[self.digitizing_bins==i])!=0])
		
		
		self.structure_function = np.asarray([])
		i=0
		for inx,val in enumerate(self.digitized_delta_mags):
			#if len(val)!=0:
			#	i+=1
			if len(val)==0:
				self.structure_function = np.append(self.structure_function, 0)
			else:
				self.structure_function = np.append(self.structure_function, ((sum([(val[i]-val[j])**2 for (i,j) in product(range(len(val)), range(len(val))) if i < j])/len(val))**(1/2))) #
		
		return None
	
	def make_structure_function_two(self, filtr = 'u'):
		"""
		Calculates the structure function for the second definition cited in the paper. 
		
		"""
		self.make_structure_function(filter=filtr) #Do calculations common to both structure functions.
		
		self.structure_function_two = np.asarray([0.74*iqr(val)/np.sqrt(len(val)-1) for val in self.digitized_delta_mags])
		
		return None
		
	def mag_v_N(self):
		"""
		Calculates the number of quasars versus magnitude. Plots the resulting array and saves to file. 
		"""
		for inx,filter in enumerate(self.filters):
			mag,_ = self.get_filter_mag(filter)
			mag,_ = self.remove_outliers(mag)
			bins = self.make_bins(mag, 0.075)
			digitized = np.digitize(mag, bins)
			binned_mag = np.asarray([mag[digitized==i] for i in range(len(digitized)) if len(mag[digitized==i])!=0])
			count = np.asarray([len(val) for val in binned_mag])
			avg_binned_mag = np.asarray([np.mean(val) for val in binned_mag])
			
			plt.plot(avg_binned_mag, count, label=str(filter) + " mag", color=self.colors[inx], linestyle = self.linestyles[inx], linewidth = self.linewidth)
		plt.title("Magnitude Distribution count")
		plt.xlabel("Magnitude")
		plt.ylabel("Number of QSOs")
		plt.legend()
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + "mag_v_N" + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
	
	def z_v_N(self):
		"""
		Calculates the number of quasars versus redshift. Plots the resulting array and saves to file
		"""
		Z,_ = self.remove_outliers(self.redshift)
		bins = self.make_bins(Z, 1/12)
		digitized = np.digitize(Z, bins)
		binned_mag = np.asarray([Z[digitized==i] for i in range(len(digitized)) if len(Z[digitized==i])!=0])
		count = np.asarray([len(val) for val in binned_mag])
		avg_binned_mag = np.asarray([np.mean(val) for val in binned_mag])
			
		plt.scatter(avg_binned_mag, count, label="redshift", s=4, color='black')
		plt.title("Redshift Distribution Count")
		plt.xlabel("Redshift (Z)")
		plt.ylabel("Number of QSOs")
		plt.legend()
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + "z_v_N" + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
	
	def plot_structure_function(self, filtr='u'):
		"""
		First calculates the structure function for a given filters and plots it. Then saves to file. 
		
		"""
		self.make_structure_function(filter=filtr)
		
		
		#fit data
		#test_func = lambda x, a, b: a + (x)*b #fitting function
		test_func = lambda x, a, b, c: a*np.exp(b*(x-c))
		pO = [1, -1, 0] #initial values for curve fitting function
		params, covar = optimize.curve_fit(test_func, self.digitized_time_lag, self.structure_function, p0 = pO, maxfev=10**6)
		
		#plot stuff
		plt.title(str(filtr) + " " + r"Magnitude difference versus time lag")
		plt.xlabel(r"rest frame Time lag (yrs)")
		plt.ylabel(r"Structure Function ($\Delta$mag)")
		plt.yscale(self.plot_scale[1])
		plt.xscale(self.plot_scale[0])
		plt.ylim(min(self.structure_function)-1, max(self.structure_function)+1)
		plt.scatter(self.digitized_time_lag, self.structure_function, s=2, label='Structure Function', color='blue'); 
		plt.plot(self.digitized_time_lag, test_func(self.digitized_time_lag, params[0], params[1], params[2]), label='Fitted Function', color='red', linewidth = self.linewidth)
		plt.plot([],[], ' ', label =  "Fitted Function params = ({0:0.3f}, {1:0.3f}, {2:0.3f})".format(params[0], params[1], params[2]))
		#plt.text(0,0, "Fitted Function params = ({0:0.3f}, {1:0.3f})".format(params[0], params[1], params[2]), wrap=True)
		plt.legend(prop={'size':9})
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + 'Structure_function_' + '(' + str(filtr) + ')' + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
	
	def plot_structure_function_two(self, filtr='u'):
		"""
		First calculates the second definition of the structure function (see paper) for a given filters and plots it. Then saves to file. 
		"""
		self.make_structure_function_two(filtr=filtr)
		
		#plot stuff
		plt.title(str(filtr) + " " + r"Magnitude difference versus time lag")
		plt.xlabel(r"rest frame Time lag (yrs)")
		plt.ylabel(r"Second Structure Function ($\Delta$mag)")
		plt.yscale(self.plot_scale[1])
		plt.xscale(self.plot_scale[0])
		plt.scatter(self.digitized_time_lag, self.structure_function_two, s=2, label='Structure Function', color='blue'); 
		#plt.text(0,0, "Fitted Function params = ({0:0.3f}, {1:0.3f})".format(params[0], params[1], params[2]), wrap=True)
		plt.legend(prop={'size':9})
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + 'second_Structure_function_' + '(' + str(filtr) + ')' + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
	
	def color_v_variability(self):
		"""
		Calculates the structure function for each filter, Takes the average of each Struc. Func. then plots it versus wavelength. Saves to file.
		"""
		var_color = []
		for i in self.filters:
			self.make_structure_function(filter=str(i))
			var_color.append(np.mean(self.structure_function))
		
		
		
		plt.xlabel('Wavelength (nm)')
		plt.ylabel('Average variability')
		plt.title('Color versus Variability')
		plt.scatter(self.wavelength, var_color, color='blue', label='Average Variability at wavelength', s=50, marker = 'o')
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + "color_v_var" + self.extension, s = 50, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()

	def redshift_v_mag(self, filter='u'):
		"""
		PLots the appar. magnitude versus redshift for the given filter. I dont think this was used in the paper. 
		"""
		
		mag1, mag2 = self.get_filter_mag(filter)
		mag1_filtered, m1_no_outlier_indices = self.remove_outliers(mag1)
		mag2_filtered, m2_no_outlier_indices = self.remove_outliers(mag2)
		
		plt.scatter(self.redshift[m1_no_outlier_indices], mag1_filtered, color='blue', label="First Observation", s=2, marker='^')
		plt.scatter(self.redshift[m2_no_outlier_indices], mag2_filtered, color='green', label = "Second Observation", s=2, marker="o")
		plt.ylabel(str(filter) + "-mag")
		plt.xlabel("redshift (z)")
		plt.title("Redshift vs. Magnitude")
		plt.legend()
		plt.tight_layout()
		if self.save:
			plt.savefig(self.savedir + "redshift_v_" + str(filter) + "_mag" + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()

	def variability_v_redshift(self, filtr = 'u', binwidth = 1/10):
		"""
		Bins redshift. Constructs structure function for all QSOs in one redshift bin. Compare structure functions
		"""
		bins = self.make_bins(self.redshift, binwidth)
		digitizing_bins = np.digitize(self.redshift, bins) #indices that bin the redshift
		
		mag1, mag2 = self.get_filter_mag(filtr)
		mag1_binned = np.asarray([mag1[digitizing_bins==i] for i in range(len(digitizing_bins))])
		mag2_binned = np.asarray([mag2[digitizing_bins==i] for i in range(len(digitizing_bins))])
		redshift_binned = np.asarray([self.redshift[digitizing_bins==i].mean() for i in range(len(digitizing_bins)) if len(self.redshift[digitizing_bins==i])>1])
		
		sfunc_avg = []
		print("Making {0} structure functions for binned redshift".format(len(bins)))
		for inx, bin in enumerate(mag1_binned):
			if len(bin)>1:
				self.make_structure_function(filter=filtr, data = [mag1_binned[inx], mag2_binned[inx]], bins=digitizing_bins==inx)
				sfunc_avg.append(self.structure_function.mean())

		plt.title("Redshift versus Variability ({0}-filter)".format(filtr))
		plt.xlabel("Redshift (Z)")
		plt.ylabel("Average Structure Function")
		plt.tight_layout()
		plt.plot(redshift_binned, sfunc_avg, linewidth = self.linewidth)
		if self.save:
			plt.savefig(self.savedir + "redshift_v_strucfunc" + "_(" + str(filtr) + ")" + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
	
	def variability_v_luminosity(self, binwidth=1/10):
		"""
		Bins absolute magnitude. constructs structure function for all QSOs in one luminosity bin. compare structure functions
		"""
		for fil in self.filters:
			bins = self.make_bins(self.M_i, binwidth)
			dig_bins = np.digitize(self.M_i, bins) #indices that bin the redshift
			
			mag1, mag2 = self.get_filter_mag(fil)
			mag1_binned = np.asarray([mag1[dig_bins==i] for i in range(len(dig_bins))])
			mag2_binned = np.asarray([mag2[dig_bins==i] for i in range(len(dig_bins))])
			lumin_binned = np.asarray([self.M_i[dig_bins==i].mean() for i in range(len(dig_bins)) if len(self.M_i[dig_bins==i])!=0])
			
			sfunc_avg = []
			print("Making {0} structure functions for binned Luminosity".format(len(bins)))
			for inx, bin in enumerate(mag1_binned):
				if len(bin)!=0:
					self.make_structure_function(filter=fil, data = [mag1_binned[inx], mag2_binned[inx]], bins = dig_bins==inx)
					sfunc_avg.append(self.structure_function.mean())
			plt.title("Luminosity versus Variability")
			plt.xlabel("Luminosity (mag)")
			plt.ylabel("Average Structure Function")
			plt.tight_layout()
			plt.plot(lumin_binned, sfunc_avg, linewidth = self.linewidth, label=str(fil) + " filter")
			plt.legend()
		if self.save:
			plt.savefig(self.savedir + "luminosity_v_strucfunc" + self.extension, dpi = self.dpi)
		if self.plot:
			plt.show()
		plt.clf()
		
	def get_filter_mag(self, filter):
		"""
		Retrieves the magnitude measurement for the given filter. 
		"""
		if filter=='u':
			mag1, mag1_err, mag2, mag2_err = self.u_PSF, self.u_err, self.u_PSF_second, self.u_PSF
		elif filter=='g':
			mag1, mag2 = self.g_PSF, self.g_PSF_second
		elif filter=='r':
			mag1, mag2 = self.r_PSF, self.r_PSF_second
		elif filter=='i':
			mag1, mag2 = self.i_PSF, self.i_PSF_second
		elif filter=='z':
			mag1, mag2 = self.z_PSF, self.z_PSF_second
		else:
			raise CustomError("ERROR: Must specify filter name (ugriz)")
		return mag1, mag2
		
	def remove_outliers(self, data, conditions=None):
		"""
		Removes outliers from the data
		"""
		if conditions==None:
			##Assume data is magnitude list
			condition = (data>0)&(data<40)
			return data[condition], condition
		## any other possible conditions?
	
	def make_bins(self, arr, frac_size):
		"""
		Generates bins with binwidth = frac_size for the array arr. 
		"""
		minm = min(arr)
		maxm = max(arr)
		return np.linspace(minm, maxm, (maxm-minm)/frac_size)

class CustomError(Exception):
	"""
	Custom exception class. 
	"""
	pass
	
	
	
	
	
	
"""
def redshift_v_variability(self):
"""
#This plot doesnt really make sense. The Struc Func is computed as a function of time lag, but this plot basically just renames the time lag to redshift, and doesnt actually 
#plot the struc func versus redshift. 
"""

for inx, fil in enumerate(self.filters):
	self.make_structure_function(filter=fil)
	self.digitizing_bins
	self.no_outlier_redshift = self.redshift[self.no_outlier_indices]
	self.redshift_binned = np.asarray([np.mean(self.no_outlier_redshift[self.digitizing_bins==i]) for i in range(len(self.digitizing_bins)) if len(self.no_outlier_redshift[self.digitizing_bins==i])!=0])


	plt.scatter(self.redshift_binned, self.structure_function, color=self.colors[inx], s=2, label='QSO ('+str(fil) + ' band)', marker=self.markers[inx])
plt.ylabel("Variability")
plt.xlabel("Redshift (Z)")
plt.title("Redshift versus variability")
plt.legend()
plt.savefig(self.savedir + "redshift_v_variability" + self.extension, dpi = self.dpi)
if self.plot:
	plt.show()
plt.clf()
"""
	