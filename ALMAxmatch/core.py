# -*- coding: utf-8 -*-

""" Exploring and searching the ALMA archive

resources:
https://nbviewer.jupyter.org/gist/keflavich/19175791176e8d1fb204

to do:
------
  -factor out ALMA source name sanitation into a private method and run on all
   query result tables
  -remove hard-coded public=False and science=False from runQueries
    -maybe the queries could always retrieve all data but we store an internal
     flag specifying those options so you can change your mind later and just
     flip the flag(s) to whatever you want
  -need some kind of check and case handling for when the queries have
   already been run but new targets are added
  -actually incorporate into the query tool class
  -make it continue to search for the next target if previous one doesn't find
   any information.
  -give message when it doesn't find any observation for the target.
  -make more methods to fully parse the frequency support column into readable
   and useable (e.g. arrays of floats) forms
    -currently done for frequency ranges for each SPW in each result row (at
     the execution block level)
  -do we want to work over the whole query result table to put all columns in
   useful forms (like the dates as datetime objects and parsing out the SPW
   frequecy ranges)?
    -if yes, brunettn thinks they should all be run automatically when the
     query is finished (like _convertDateColumnsToDatetime is now)
  -add description somewhere that when querying a region, the targets added
   must be tuples with (coordinates, radius) specified like the first two
   parameters of Alma.query_region
  -ideas for better name matching
    -"N" instead of full "NGC" in name is sometimes used
    -search for substrings for name matching
    -search in NED for coords and pass that to ALMA
"""

from astropy import units as u
from astropy.table import join, setdiff, unique, vstack
from astroquery.alma import Alma
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
import numpy as np
import string
import matplotlib.pyplot as plt

# fix Python SSL errors when downloading using the https
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

# Today's date (handy for determining public data) 
now = np.datetime64('now')

class archiveSearch:
    """
    Data attributes:
    ----------------
    -targets
    -queryResults
    -queryResultsNoNED
    -queryResultsNoNEDz
    -uniqueBands
    -isObjectQuery
    """
    def __init__(self, targets):
        self.targets = dict()
        self.isObjectQuery = dict()
        if type(targets) is list:
            for i in range(len(targets)):
                self.addTarget(targets[i])
        else:
            self.addTarget(targets)

        self.queryResults = None
        self.queryResultsNoNED = None
        self.queryResultsNoNEDz = None

    def addTarget(self, target):
        """Add target to object's dictionary of targets.

        Will accept a string indicating a target name or a tuple of
        (coordinates, radius) indicating a region to search. Tuple entries can
        be either strings or an astropy.coordinates object for the coordinates
        entry and an astropy.units.Quantity object for the radius entry.
        """
        targetType = type(target)
        if targetType == str:
            self.targets[target] = target
            self.isObjectQuery[target] = True
        elif targetType == tuple:
            if type(target[0]) == SkyCoord:
                targetStr = 'coord=({:} {:}) radius={:}'.format(target[0].ra,target[0].dec,target[-1])
            else:
                targetStr = 'coord={:} radius={:}'.format(*target)
            self.targets[targetStr] = target
            self.isObjectQuery[targetStr] = False
        else:
            raise TypeError('Cannot work with targets '
                            +'of type {:}'.format(targetType))

    def runQueries(self, public=False, science=False, **kwargs):
        """Run queries on list of targets saved in archiveSearch object.

        Loops through the list of targets or regions and runs
        astroquery.alma.query_object[region] on each. The results are stored in
        the list archiveSearch.queryResults.

        Also does some work on the result tables to put some of their data into
        more useful forms. This includes:
          -converting the 'Release' and 'Observation' data columns from strings
           to np.datetime64 objects
        """
        self.queryResults = dict()
        self.invalidName=list()
        for target in self.targets:
            if self.isObjectQuery[target] == True: # for individual sources
                try: 
                    self.queryResults[target] = Alma.query_object(target,
                                                                 public=public,
                                                                 science=science,
                                                                  **kwargs)
                    self.isObjectQuery[target] = True
                except ValueError:
                    self.invalidName.append(target)
                    print("Invalid Name '"+target+"'")
            
            else: # for querying regions
                results = Alma.query_region(*self.targets[target],
                                                public=public,
                                                science=science,
                                                **kwargs)
                self.queryResults[target] = results
        for key in self.invalidName:
            self.targets.pop(key)
        self._convertDateColumnsToDatetime()

    def _convertDateColumnsToDatetime(self):
        """Convert archive query result dates to np.datetime64 objects.

        Columns like 'Release date' and 'Observation date' in the archive
        query results tables are initially strings. This converts those
        columns, for all targets, into np.datetime64 objects so they are more
        useful.

        The underscore at the beginning of the name indicates this is intended
        to be used by the archiveSearch object internally so no guarantees are made
        if you try using it manually.
        """
        for target in self.targets:
            relCol = self.queryResults[target]['Release date']
            obsCol = self.queryResults[target]['Observation date']
            for i in range(len(relCol)):
                relCol[i] = np.datetime64(relCol[i])
                obsCol[i] = np.datetime64(obsCol[i])
            self.queryResults[target]['Release date'] = relCol
            self.queryResults[target]['Observation date'] = obsCol

    def observedBands(self):
        """Save unique bands into archiveSearch object.
        """
        self.uniqueBands = dict()
        for tar in self.targets:
            self.uniqueBands[tar] = np.unique(self.queryResults[tar]['Band'])

    def parseFrequencyRanges(self):
        """Parses observed frequency ranges into something more useable.

        Loops through the list of targets and then through each query result
        row pulling out the SPW frequency ranges stored in the query result
        column 'Frequency support.' A new column is then added to the target
        query result table called 'Frequency ranges' where lists of astropy
        quantity 2-tuples are stored that give the maximum and minimum
        frequency in each SPW for each row (i.e. execution block).

        The new column is easy to read by people and is in a form where math
        can be done with the frequencies. Each frequency is an astropy
        float quantity with units.
        """
        for tar in self.targets:
            targetFreqRanges = list()
            freqUnit = self.queryResults[tar]['Frequency support'].unit
            for i in range(len(self.queryResults[tar])):
                freqStr = self.queryResults[tar]['Frequency support'][i]
                freqStr = freqStr.split('U')
                rowFreqRanges = list()
                for j in range(len(freqStr)):
                    freqRange = freqStr[j].split(',')
                    freqRange = freqRange[0].strip(' [')
                    freqRange = freqRange.split('..')
                    freqRange[1] = freqRange[1].strip(string.ascii_letters)
                    freqRange = np.array(freqRange, dtype='float')
                    rowFreqRanges.append(freqRange)
                targetFreqRanges.append(rowFreqRanges)
            self.queryResults[tar]['Frequency ranges'] = targetFreqRanges
            self.queryResults[tar]['Frequency ranges'].unit = freqUnit

    def dumpSearchResults(self, target_data, bands,
                          unique_public_circle_parameters=False,
                          unique_private_circle_parameters=False): 
        print("Total observations: {0}".format(len(target_data)))
        print( "Unique bands: ", bands)
        for band in bands:
            print("BAND {0}".format(band))
            privrows = sum((target_data['Band']==band) & (target_data['Release date']>now))
            pubrows  = sum((target_data['Band']==band) & (target_data['Release date']<=now))
            print("PUBLIC:  Number of rows: {0}.  Unique pointings: {1}".format(pubrows, len(unique_public_circle_parameters[band])))
            print("PRIVATE: Number of rows: {0}.  Unique pointings: {1}".format(privrows, len(unique_private_circle_parameters[band])))

    def printQueryResults(self, max_lines=None, max_width=None, show_name=True,
                          show_unit=None, show_dtype=False, align=None):
        """Print formatted string representation of the query result table(s).

        This method directly pipes its arguments into the
        astropy.table.Table.pprint method, so please see the documentation for
        that method for descriptions of each argument. If multiple fields were
        queried then this method will loop over each field, running pprint for
        the corresponding results table.
        """
        for target in self.targets:
            print(target)
            self.queryResults[target].pprint(max_lines=max_lines,
                                             max_width=max_width,
                                             show_name=show_name,
                                             show_unit=show_unit,
                                             show_dtype=show_dtype,
                                             align=align)
            print('\n\n')

    def formatQueryResults(self, max_lines=None, max_width=None,
                           show_name=True, show_unit=None, show_dtype=False,
                           html=False, tableid=None, align=None,
                           tableclass=None):
        """Return a list of lines for the formatted string representation of
        the query result table(s).

        This method directly pipes its arguments into the
        astropy.table.Table.pformat method, so please see the documentation for
        that method for descriptions of each argument. If multiple fields were
        queried then this method will loop over each field, running pformat for
        the corresponding results table.
        """
        lines = list()
        for target in self.targets:
            lines.append(target)
            lines.extend(self.queryResults[target].pformat(max_lines=max_lines,
                                                           max_width=max_width,
                                                           show_name=show_name,
                                                           show_unit=show_unit,
                                                         show_dtype=show_dtype,
                                                           html=html,
                                                           tableid=tableid,
                                                           align=align,
                                                        tableclass=tableclass))
            lines.append('')
            lines.append('')
        return lines

    # def largeSkyQueryWithLines(self, restFreqs, redshiftRange=(0, 1000), line_names="", **kwargs):
    #      """Running search on large search areas.

    #     Accepts a list of lines and a redshift range, searches the ALMA archive for line observations

    #     Parameters
    #     ----------

    #     restFreqs : array_like
    #         The spectral line rest frequencies to search the query results for

    #     redshiftRange : 2 element array_like (lowz, highz), optional
    #         A 2-element list, tuple, etc. defining the lower and upper limits
    #         of the object redshifts (in that order) to be searched for. The 
    #         restFreqs will be shifted using this range to only find 
    #         observations that have spectral coverage in that redshift range. 
    #         Default is to search from z=0 to 1000 (i.e. all redshifts).

    #     All arguments are passed to astroquery.alma.Alma.query_object except
    #     frequency, which cannot be specified here since it is used to limit 
    #     the query to frequencies that could contain the lines in the specified
    #     redshift range.

    #     archiveSearch.queryResults will contain an astropy table with all observations
    #     that match a NED object name and have a redshift, with flags for each
    #     line specifying if the spectral coverage includes the line frequency for
    #     the object's redshift.

    #     archiveSearch.queryResultsNoNED will contain an astropy table with all
    #     observations that did not have a match in NED, based on name.

    #     archiveSearch.queryResultsNoNEDz will contain an astropy table with all
    #     observations that match a NED object name but do not have a redshift.
    #     """
    #     pass
        
    def _observedFreq(self, restFreq, z):
        """Return observed frequency according to nu_0 / nu = 1 + z."""
        return restFreq/(1+z)

    def _lineObserved(self, target_frequency, observed_frequency_ranges, linename=""):
        """Loop through the observed spectral windows to check if 
            target_frequency is covered by spectral setup"""
        
        # Initialize boolean line observed flag array (i.e., True = line frequency in archive spw coverage)
        lineObserved = []
        
        # loop over spectral window frequencies for each observation
        for spw in observed_frequency_ranges:
            # if observed frequency is in spw, flag as True and break inner loop (move on to next observation) 
            if spw[0] <= target_frequency <= spw[-1]:
                print(linename,"observed frequency", target_frequency, "GHz",
                          "in range", spw[0], "-", spw[-1])
                lineObserved.append(True) # line IS observed
            else:
                lineObserved.append(False) # line NOT observed
            
        # Boolean line observed flag for each observation 
        if True in lineObserved:
            return True
        else:
            return False

    def runQueriesWithLines(self, restFreqs, redshiftRange=(0, 1000), line_names="", **kwargs):
        """Run queries for spectral lines on targets saved in archiveSearch object.

        Parameters
        ----------

        restFreqs : array_like
            The spectral line rest frequencies to search the query results for

        redshiftRange : 2 element array_like (lowz, highz), optional
            A 2-element list, tuple, etc. defining the lower and upper limits
            of the object redshifts (in that order) to be searched for. The 
            restFreqs will be shifted using this range to only find 
            observations that have spectral coverage in that redshift range. 
            Default is to search from z=0 to 1000 (i.e. all redshifts).

        All arguments are passed to astroquery.alma.Alma.query except
        frequency, which cannot be specified here since it is used to limit 
        the query to frequencies that could contain the lines in the specified
        redshift range.

        archiveSearch.queryResults will contain an astropy table with all observations
        that match a NED object name and have a redshift, with flags for each
        line specifying if the spectral coverage includes the line frequency for
        the object's redshift.

        archiveSearch.queryResultsNoNED will contain an astropy table with all
        observations that did not have a match in NED, based on name.

        archiveSearch.queryResultsNoNEDz will contain an astropy table with all
        observations that match a NED object name but do not have a redshift.
        """
        if 'frequency' in kwargs:
            msg = '"frequency" cannot be passed to runQueriesWithLines'
            raise ValueError(msg)

        if (len(line_names) != len(restFreqs)) and (line_names != ""):
                raise TypeError('line_names: expecting either empty string [default] or list of strings that is length={:}'.format(len(restFreqs)))

        restFreqs = np.array(restFreqs)
        restFreqs.sort()

        redshiftRange = np.array(redshiftRange)
        redshiftRange.sort()

        # define frequency range from lines and redshifts
        lowFreq = self._observedFreq(restFreqs[0], redshiftRange[1])
        highFreq = self._observedFreq(restFreqs[-1], redshiftRange[0])
        freqLimits = '{:} .. {:}'.format(lowFreq, highFreq)

        self.runQueries(frequency=freqLimits, **kwargs)

        self.parseFrequencyRanges()

        self.queryResultsNoNED = dict()
        self.queryResultsNoNEDz = dict()

        for target in self.targets:
            if len(self.queryResults[target])>0: # only do this for targets with ALMA results

                currTable = self.queryResults[target]

                # sanitize ALMA source names
                safeNames = currTable['Source name']
                safeNames = np.char.replace(safeNames, b' ', b'')
                safeNames = np.char.replace(safeNames, b'_', b'')
                safeNames = np.char.upper(safeNames)
                currTable['ALMA sanitized source name'] = safeNames

                uniqueALMAnames = np.unique(currTable['ALMA sanitized source name'])

                # query NED for object redshifts
                nedResult = list()
                for sourceName in uniqueALMAnames:
                    try:
                        nedSearch = Ned.query_object(sourceName)
                        nedSearch['ALMA sanitized source name'] = sourceName
                        # doing this prevents vstack warnings
                        nedSearch.meta = None
                        nedResult.append(nedSearch)
                    except Exception:
                        pass
                nedResult = vstack(nedResult)

                # store rows without matching name in NED
                self.queryResultsNoNED[target] = setdiff(currTable, nedResult,
                                                         keys='ALMA sanitized source name')

                # remove rows without redshifts in NED
                blankZinds = nedResult['Redshift'].mask.nonzero()
                blankZnames = nedResult['ALMA sanitized source name'][blankZinds]
                nedResult.remove_rows(blankZinds)

                # store rows with matching name in NED but no z
                # (this seems like a dumb approach)
                blankZinds = list()
                for i,row in enumerate(currTable):
                    if row['ALMA sanitized source name'] in blankZnames:
                        blankZinds.append(i)
                self.queryResultsNoNEDz[target] = currTable[blankZinds]

                # rectify this naming difference between NED and ALMA
                nedResult.rename_column('DEC', 'Dec')

                # keep redshifts, positions too if we wanna check later
                nedResult.keep_columns(['Object Name', 'RA', 'Dec', 'Redshift',
                                        'ALMA sanitized source name'])

                # join NED redshift table and ALMA archive table based on name
                ALMAnedResults = join(currTable, nedResult,
                                      keys='ALMA sanitized source name')

                # tidy up column names
                ALMAnedResults.rename_column('Source name', 'ALMA source name')
                ALMAnedResults.rename_column('RA_1', 'ALMA RA')
                ALMAnedResults.rename_column('Dec_1', 'ALMA Dec')
                ALMAnedResults.rename_column('Object Name', 'NED source name')
                ALMAnedResults.rename_column('RA_2', 'NED RA')
                ALMAnedResults.rename_column('Dec_2', 'NED Dec')
                ALMAnedResults.rename_column('Redshift', 'NED Redshift')

                # mark flags if spw is on line (initialized to False)
                lineObserved = np.zeros((len(ALMAnedResults), len(restFreqs)),
                                         dtype=bool)


                for i, row in enumerate(ALMAnedResults):

                    # target redshift
                    z = row['NED Redshift']
                    
                    # observed frequencies
                    observed_frequencies = [self._observedFreq(rf, row['NED Redshift']) for rf in restFreqs]

                    if line_names == "":
                        line_names = ['Line{:}'.format(i) for i in range(len(restFreqs))]

                    # loop over the target lines, return a boolean flag array and add it to astropy table
                    for j, (observed_frequency, linename) in enumerate(zip(observed_frequencies,line_names)):
                        lineObserved[i, j]=self._lineObserved(target_frequency=observed_frequency
                                                                    , observed_frequency_ranges=row['Frequency ranges']
                                                                    , linename=linename)

                for i in range(len(restFreqs)):
                    ALMAnedResults[line_names[i]] = lineObserved[:, i]

                # remove rows which have no lines covered
                lineCount = np.array(ALMAnedResults[line_names[0]], dtype=int)
                for i in range(1, len(restFreqs)):
                    lineCount += np.array(ALMAnedResults[line_names[i]], dtype=int)
                noLinesInds = np.where(lineCount == 0)
                ALMAnedResults.remove_rows(noLinesInds)

                self.queryResults[target] = ALMAnedResults


if __name__ == "__main__":
    # region query with line search
    if True:
        target = ('12h26m32.1s 12d43m24s', '6deg')
        myarchiveSearch = archiveSearch(target)
        mySurvey.runQueriesWithLines([113.123337, 230.538],
                                     redshiftRange=(0, 0.1),
                                     science=True)
        print(len(mySurvey.queryResults['coord=12h26m32.1s 12d43m24s radius=6deg']))
        print(mySurvey.queryResultsNoNED['coord=12h26m32.1s 12d43m24s radius=6deg'])
        print(mySurvey.queryResultsNoNEDz['coord=12h26m32.1s 12d43m24s radius=6deg'])

    # region query
    if False:
        target = ('12h26m32.1s 12d43m24s', '30arcmin')
        mySurvey = survey(target)
        mySurvey.runQueries()
        #mySurvey.observedBands()
        #mySurvey.parseFrequencyRanges()
        mySurvey.printQueryResults()

    # object name query
    if False:
        targets = ['Arp 220', '30 Doradus']
        print(targets)
        print("--------------")

        mySurvey = survey(targets)
        mySurvey.runQueries()
        mySurvey.observedBands()
        mySurvey.parseFrequencyRanges()
        print(mySurvey.targets)
        print(mySurvey.uniqueBands)
        mySurvey.printQueryResults()
        lines = mySurvey.formatQueryResults(max_lines=-1, max_width=-1)
        with open('survey_out.txt', 'w') as f:
            for line in lines:
                f.write(line+'\n')
