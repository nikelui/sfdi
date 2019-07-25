from numpy import array,squeeze

def csvread(filename,arr=False,delimiter=','):
    """Function to read custom CSV data

syntax: (data,header) = csvread(filename[,delimiter])
- filename is a string with the file path
- arr is a boolean. If false it reads in a list, if true in a numpy array
- delimiter[optional] is a string to specify a separator different from the default ','

The CSV format is the following:
- any number of header lines starting with a '#' character (or % for Matlab compatibility)
- any number of arrays, one per line
The header is saved in a list of strings (one per line)
The data is saved in a matrix by rows, with the float type."""
    header = []
    data = []
    f = open(filename,'r')
    for line in f:
        if line[0] in '#%':
            header.append(line[1:].strip()) # Save to header
        elif (len(line.strip()) == 0):
            continue # skip empty lines
        else:
            row = [float(x) for x in line.split(delimiter)]
            data.append(row)
    f.close()
    if arr:
        data = array(data)
        data = squeeze(data)
    return(data,header)
    
def csvwrite(filename,data,header=[]):
    """Function to write custom CSV data

The CSV format is the following:
- any number of header lines starting with a '#' character
- any number of arrays, one per line
The header must be passed as a list of string (one string per line)
The data must be passed in matrix form (one row per array of data)."""
    f = open(filename,'w')
    if(len(header) > 0):
        for line in header:
            f.write('#'+line+'\n')
    for row in data:
        row = [str(x) for x in row] # convert to string to use the join() method
        f.write(','.join(row)+'\n') # The default separator should be ',' anyways
    f.close()