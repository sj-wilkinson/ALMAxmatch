# -*- coding: utf-8 -*-

""" Exploring and searching the ALMA archive

resources:
https://nbviewer.jupyter.org/gist/keflavich/19175791176e8d1fb204

to do:
------
  -factor out ALMA source name sanitation into a private method and run on all
   query result tables
  -remove hard-coded public=False and science=False from runTargetQuery
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

from astropy.table import join, setdiff, vstack
from astroquery.alma import Alma
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
import numpy as np
import string

# fix Python SSL errors when downloading using the https
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

class archiveSearch:
    """ALMA archive search with cross matching against NED.

    Parameters
    ----------
    targets : array_like, optional
        A sequence of strings and/or tuples specifying source names and regions
        with which to query the ALMA archive, respectively. Region tuples must
        consist of (coordinates, radius) where the coordinates element can be
        either a string or an astropy.coordinates object and the radius element
        can be either a string or an astropy.units.Quantity object.

    Attributes
    ----------
    targets : array_like
        See parameter description.
    queryResults : dict
        Results from querying the ALMA archive, with the targets queried as the
        keys and astropy tables containing the observation information as the
        corresponding values. If one of the "WithLines" methods was run, only
        targets that with ALMA observations containing spectral windows that
        overlapped the requested line(s) and had matches in NED with redshifts
        appear here.
    queryResultsNoNED : dict
        Same as queryResults but only containing the targets that had data in
        the ALMA archive but did not have a match in NED.
    queryResultsNoNEDz : dict
        Same as queryResults but only containing the targets that had data in
        the ALMA archive and matched a source in NED but NED had no redshift
        information.
    uniqueBands : dict
        The unique bands in query results organized with each queried target
        as keys and the unique bands as the corresponding values. The bands are
        stored in astropy.table.column objects.
    isObjectQuery : dict
        Booleans that indicate whether each queried target is a source name
        (True) or a region (False). Targets are the keys and the boolean flags
        are the values.
    invalidName : list strs
        List of source name strings specifying targets that did not return any
        results from the ALMA archive query.
    """

    def __init__(self, targets=None):
        self.isObjectQuery = dict()

        self.targets = dict()
        if type(targets) is list:
            for i in range(len(targets)):
                self.addTarget(targets[i])
        elif targets != None:
            self.addTarget(targets)

        self.invalidName = list()

        self.queryResults = dict()
        self.queryResultsNoNED = dict()
        self.queryResultsNoNEDz = dict()

        self.uniqueBands = dict()

    def runPayloadQuery(self, payload, **kwargs):
        """Perform a generic ALMA archive query with user-specified payload.

        Parameters
        ----------
        payload : dict
            A dictionary of payload keywords that are accepted by the ALMA
            archive system. You can look these up by examining the forms at
            http://almascience.org/aq. Passed to `astroquery.alma.Alma.query`.
        public : bool
            Return only publicly available datasets?
        science : bool
            Return only data marked as "science" in the archive?
        kwargs : dict
            Passed to `astroquery.alma.Alma.query`.
        """
        results = Alma.query(payload, **kwargs)
        self.queryResults = results
        self._convertDateColumnsToDatetime()


    def runPayloadQueryWithLines(self, restFreqs, payload=None,
                                 redshiftRange=(0, 1000), lineNames=[],
                                 **kwargs):
        """Run query for spectral lines with user-specified payload.

        Parameters
        ----------
        restFreqs : sequence of floats
            The spectral line rest frequencies to search the query results for.
        payload : dict
            A dictionary of payload keywords that are accepted by the ALMA
            archive system. You can look these up by examining the forms at
            http://almascience.org/aq. Passed to `astroquery.alma.Alma.query`.
        redshiftRange : sequence of floats, optional
            A two-element sequence defining the lower and upper limits of the
            object redshifts (in that order) to be searched for. The restFreqs
            will be shifted using this range to only find observations that
            have spectral coverage in that redshift range. Default is to search
            0 <= z <= 1000 (i.e. all redshifts).
        lineNames : sequence of strs, optional
            A sequence of strings containing names for each spectral line to
            be searched for that will be used as column names in the results
            table. This must be the same length as restFreqs. Default is to
            name lines like "Line0", "Line1", "Line2", etc.
        public : bool
            Return only publicly available datasets?
        science : bool
            Return only data marked as "science" in the archive?
        kwargs : dict
            Passed to `astroquery.alma.Alma.query` except frequency, which
            cannot be specified here since it is used to limit the query to
            frequencies that could contain the lines in the specified redshift
            range.
        """
        if 'frequency' in kwargs:
            msg = '"frequency" cannot be passed to runPayloadQueryWithLines'
            raise ValueError(msg)

        lineNames = np.array(lineNames)
        if (len(lineNames) != len(restFreqs) and len(lineNames) != 0):
            msg = 'length mismatch between '
            msg += '"restFreqs" ({:}) '.format(len(restFreqs))
            msg += 'and "lineNames" ({:})'.format(len(lineNames))
            raise ValueError(msg)

        restFreqs = np.array(restFreqs)

        redshiftRange = np.array(redshiftRange)
        redshiftRange.sort()

        # define frequency range from lines and redshifts
        lowFreq = self._observedFreq(np.sort(restFreqs)[0], redshiftRange[1])
        highFreq = self._observedFreq(np.sort(restFreqs)[-1], redshiftRange[0])
        freqLimits = '{:} .. {:}'.format(lowFreq, highFreq)


        # ALMA archive keyword payload
        if payload is None:
            payload = {}
        payload['frequency'] = freqLimits

        self.runPayloadQuery(payload=payload, **kwargs)

        self.parseFrequencyRanges() # THIS NEEDS FIXING.

        # sanitize ALMA source names
        safeNames = self.queryResults['Source name']
        safeNames = np.char.replace(safeNames, b' ', b'')
        safeNames = np.char.replace(safeNames, b'_', b'')
        safeNames = np.char.upper(safeNames)
        self.queryResults['ALMA sanitized source name'] = safeNames

        uniqueALMAnames = np.unique(self.queryResults['ALMA sanitized source name'])

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
        queryResultsNoNED = setdiff(self.queryResults, nedResult,
                                                 keys='ALMA sanitized source name')

        # remove rows without redshifts in NED
        blankZinds = nedResult['Redshift'].mask.nonzero()
        blankZnames = nedResult['ALMA sanitized source name'][blankZinds]
        nedResult.remove_rows(blankZinds)


        # store rows with matching name in NED but no z
        # (this seems like a dumb approach)
        blankZinds = list()
        for i,row in enumerate(self.queryResults):
            if row['ALMA sanitized source name'] in blankZnames:
                blankZinds.append(i)
        queryResultsNoNEDz = self.queryResults[blankZinds]

        # remove rows where redshift not in range
        outofrangeZinds = []
        for i,row in enumerate(nedResult):
            if (redshiftRange[0] <= row['Redshift'] <= redshiftRange[1]) == False:
                outofrangeZinds.append(i)
        nedResult.remove_rows(outofrangeZinds)

        # rectify this naming difference between NED and ALMA
        nedResult.rename_column('DEC', 'Dec')

        # keep redshifts, positions too if we wanna check later
        nedResult.keep_columns(['Object Name', 'RA', 'Dec', 'Redshift',
                                'ALMA sanitized source name'])

        # generate a human readable, pythonic column of spectral window frequency ranges
        self.parseFrequencyRanges()

        # join NED redshift table and ALMA archive table based on name
        ALMAnedResults = join(self.queryResults, nedResult,
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

            if len(lineNames) == 0:
                lineNames = ['Line{:}'.format(i) for i in range(len(restFreqs))]

            # loop over the target lines, return a boolean flag array and add it to astropy table
            for j, (observed_frequency, linename) in enumerate(zip(observed_frequencies,lineNames)):
                lineObserved[i, j]=self._lineObserved(lineFreq=observed_frequency
                                                            , spwFreqLims=row['Frequency ranges'])

        # add flag columns to array
        for i in range(len(restFreqs)):
                    ALMAnedResults[lineNames[i]] = lineObserved[:, i]

        # remove rows which have no lines covered
        lineCount = np.array(ALMAnedResults[lineNames[0]], dtype=int)
        for i in range(1, len(restFreqs)):
            lineCount += np.array(ALMAnedResults[lineNames[i]], dtype=int)
        noLinesInds = np.where(lineCount == 0)
        ALMAnedResults.remove_rows(noLinesInds)

        self.queryResults = ALMAnedResults

    def runTargetQuery(self, public=False, science=False, **kwargs):
        """Run queries on list of targets.

        Parameters
        ----------
        public : bool
            Return only publicly available datasets?
        science : bool
            Return only data marked as "science" in the archive?
        kwargs : dict
            Passed to `astroquery.alma.Alma.query_object` or
            `astroquery.alma.Alma.query_region`.

        Also does some work on the result tables to put data into more useful
        forms. This includes:

          * converting the 'Release' and 'Observation' data columns from
            strings to np.datetime64 objects
        """
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

    def runTargetQueryWithLines(self, restFreqs, redshiftRange=(0, 1000),
                                lineNames=[], **kwargs):
        """Run queries for spectral lines on list of targets.

        Parameters
        ----------
        restFreqs : sequence of floats
            The spectral line rest frequencies to search the query results for.
        redshiftRange : sequence of floats, optional
            A two-element sequence defining the lower and upper limits of the
            object redshifts (in that order) to be searched for. The restFreqs
            will be shifted using this range to only find observations that
            have spectral coverage in that redshift range. Default is to search
            0 <= z <= 1000 (i.e. all redshifts).
        lineNames : sequence of strs, optional
            A sequence of strings containing names for each spectral line to
            be searched for that will be used as column names in the results
            table. This must be the same length as restFreqs. Default is to
            name lines like "Line0", "Line1", "Line2", etc.
        public : bool
            Return only publicly available datasets?
        science : bool
            Return only data marked as "science" in the archive?
        kwargs : dict
            Passed to `astroquery.alma.Alma.query_object` except frequency,
            which cannot be specified here since it is used to limit the query
            to frequencies that could contain the lines in the specified
            redshift range.
        """
        if 'frequency' in kwargs:
            msg = '"frequency" cannot be passed to runTargetQueryWithLines'
            raise ValueError(msg)

        lineNames = np.array(lineNames)
        if (len(lineNames) != len(restFreqs) and len(lineNames) != 0):
            msg = 'length mismatch between '
            msg += '"restFreqs" ({:})'.format(len(restFreqs))
            msg += ' and "lineNames" ({:})'.format(len(lineNames))
            raise ValueError(msg)

        restFreqs = np.array(restFreqs)

        redshiftRange = np.array(redshiftRange)
        redshiftRange.sort()

        # define frequency range from lines and redshifts
        lowFreq = self._observedFreq(np.sort(restFreqs)[0], redshiftRange[1])
        highFreq = self._observedFreq(np.sort(restFreqs)[-1], redshiftRange[0])
        freqLimits = '{:} .. {:}'.format(lowFreq, highFreq)

        self.runTargetQuery(frequency=freqLimits, **kwargs)

        self.parseFrequencyRanges()

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

                    if len(lineNames) == 0:
                        lineNames = ['Line{:}'.format(i) for i in range(len(restFreqs))]

                    # loop over the target lines, return a boolean flag array and add it to astropy table
                    for j, (observed_frequency, linename) in enumerate(zip(observed_frequencies,lineNames)):
                        lineObserved[i, j]=self._lineObserved(lineFreq=observed_frequency
                                                                    , spwFreqLims=row['Frequency ranges'])

                for i in range(len(restFreqs)):
                    ALMAnedResults[lineNames[i]] = lineObserved[:, i]

                # remove rows which have no lines covered
                lineCount = np.array(ALMAnedResults[lineNames[0]], dtype=int)
                for i in range(1, len(restFreqs)):
                    lineCount += np.array(ALMAnedResults[lineNames[i]], dtype=int)
                noLinesInds = np.where(lineCount == 0)
                ALMAnedResults.remove_rows(noLinesInds)

                self.queryResults[target] = ALMAnedResults

    def addTarget(self, target):
        """Add target to archiveSearch object.

        Parameters
        ----------
        target : str or tuple
            Target to query the ALMA archive for. Can be either a string
            indicating a source name (e.g. 'M87') or a tuple indicating a
            region to search consisting of (coordinates, radius). The
            coordinates element can be either a string or an
            astropy.coordinates object and the radius element can be either a
            string or an astropy.units.Quantity object.
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

    def _convertDateColumnsToDatetime(self):
        """Convert archive query result dates to np.datetime64 objects.

        Columns like 'Release date' and 'Observation date' in the archive
        query results tables are initially strings. This converts those
        columns, for all targets, into np.datetime64 objects so they are more
        useful.
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
        for tar in self.targets:
            self.uniqueBands[tar] = np.unique(self.queryResults[tar]['Band'])

    def parseFrequencyRanges(self):
        """Parses observed frequency ranges into something more useable.

        Loops through the list of targets and then through each query result
        row pulling out the spectral window (SPW) frequency ranges stored in
        the query result column 'Frequency support'. A new column is then added
        to the target query result table called 'Frequency ranges' where lists
        of astropy quantity 2-tuples are stored that give the maximum and
        minimum frequency in each SPW for each row (i.e. execution block).

        The new column is easy to read by people and is in a form where math
        can be done with the frequencies. Each frequency is an astropy float
        quantity with units.
        """
        if len(self.targets)>=1:    
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
        
        else:
            """parse frequency ranges for payload query"""
            targetFreqRanges = list()
            freqUnit = self.queryResults['Frequency support'].unit
            for i in range(len(self.queryResults)):
                freqStr = self.queryResults['Frequency support'][i]
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
            self.queryResults['Frequency ranges'] = targetFreqRanges
            self.queryResults['Frequency ranges'].unit = freqUnit

    def dumpSearchResults(self, target_data, bands,
                          unique_public_circle_parameters=False,
                          unique_private_circle_parameters=False):
        now = np.datetime64('now')
        print("Total observations: {0}".format(len(target_data)))
        print( "Unique bands: ", bands)
        for band in bands:
            print("BAND {0}".format(band))
            privrows = sum((target_data['Band']==band) & (target_data['Release date']>now))
            pubrows  = sum((target_data['Band']==band) & (target_data['Release date']<=now))
            print("PUBLIC:  Number of rows: {0}.  Unique pointings: {1}".format(pubrows, len(unique_public_circle_parameters[band])))
            print("PRIVATE: Number of rows: {0}.  Unique pointings: {1}".format(privrows, len(unique_private_circle_parameters[band])))

    def printQueryResults(self, **kwargs):
        """Print formatted string representation of the query result table(s).

        Parameters
        ----------
        kwargs : dict
            Passed to `astropy.table.Table.pprint`.

        If multiple fields were queried then this method will loop over each
        field, running pprint for the corresponding results table.
        """
        for target in self.targets:
            print(target)
            self.queryResults[target].pprint(**kwargs)
            print('\n\n')

    def formatQueryResults(self, **kwargs):
        """Return the formatted string form of the query result table(s).

        Parameters
        ----------
        kwargs : dict
            Passed to `astropy.table.Table.pformat`

        Returns
        -------
        list
            List of strings containing each line of the formatted string form
            of the query result table(s).

        If multiple fields were queried then this method will loop over each
        field, running pformat for the corresponding results table.
        """
        lines = list()
        for target in self.targets:
            lines.append(target)
            lines.extend(self.queryResults[target].pformat(**kwargs))
            lines.append('')
            lines.append('')
        return lines
        
    def _observedFreq(self, restFreq, z):
        """Return observed frequency according to nu_0 / nu = 1 + z.

        Parameters
        ----------
        restFreq : float
            Rest frequency of line to calculate observed frequency for.
        z : float
            Redshift of observed object.

        Returns
        -------
        float
            `restFreq` / (1 + `z`)
        """
        return restFreq/(1+z)

    def _lineObserved(self, lineFreq, spwFreqLims):
        """Return whether target frequency lies within frequency ranges.

        Parameters
        ----------
        lineFreq : float
            Spectral line frequency to check for within `spwFreqLims`.
        spwFreqLims : array_like
            A sequence of two-element sequences
            [(low freq1, high freq1), (low freq2, high freq2), ...] that
            define frequency ranges that are checked whether they contain
            `lineFreq`.

        Returns
        -------
        bool
            Whether `lineFreq` lies within any frequency range in
            `spwFreqLims`.
        """
        lineObserved = False

        for spw in spwFreqLims:
            if spw[0] <= lineFreq <= spw[1]:
                lineObserved = True
                break

        return lineObserved


if __name__ == "__main__":
    # region query with line search
    if True:
        target = ('12h26m32.1s 12d43m24s', '6deg')
        myarchiveSearch = archiveSearch(target)
        mySurvey.runTargetQueryWithLines([113.123337, 230.538],
                                     redshiftRange=(0, 0.1),
                                     science=True)
        print(len(mySurvey.queryResults['coord=12h26m32.1s 12d43m24s radius=6deg']))
        print(mySurvey.queryResultsNoNED['coord=12h26m32.1s 12d43m24s radius=6deg'])
        print(mySurvey.queryResultsNoNEDz['coord=12h26m32.1s 12d43m24s radius=6deg'])

    # region query
    if False:
        target = ('12h26m32.1s 12d43m24s', '30arcmin')
        mySurvey = survey(target)
        mySurvey.runTargetQuery()
        #mySurvey.observedBands()
        #mySurvey.parseFrequencyRanges()
        mySurvey.printQueryResults()

    # object name query
    if False:
        targets = ['Arp 220', '30 Doradus']
        print(targets)
        print("--------------")

        mySurvey = survey(targets)
        mySurvey.runTargetQuery()
        mySurvey.observedBands()
        mySurvey.parseFrequencyRanges()
        print(mySurvey.targets)
        print(mySurvey.uniqueBands)
        mySurvey.printQueryResults()
        lines = mySurvey.formatQueryResults(max_lines=-1, max_width=-1)
        with open('survey_out.txt', 'w') as f:
            for line in lines:
                f.write(line+'\n')
