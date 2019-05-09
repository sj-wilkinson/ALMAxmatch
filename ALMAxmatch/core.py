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
from astroquery.utils import commons
from astropy.coordinates import SkyCoord, Angle
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
    targets : list of strs and/or lists, optional
        A list containing strings and/or lists specifying source names and
        regions with which to query the ALMA archive, respectively. Region
        lists must consist of [coordinates, radius] where the coordinates
        element can be either a string or an astropy.coordinates object and the
        radius element can be either a string or an astropy.units.Quantity
        object. This is mutually exclusive with the `allSky` argument such that
        if `allSky` is False then this cannot be None and if `allSky` is True
        this must be None.
    allSky : bool, optional
        If set to True, the archive query will be done over the entire sky
        (i.e. no source names or regions limiting where on the sky data was
        taken). This is mutually exclusive with the `targets` argument such
        that if `targets` is None then this must be True and if `targets` is
        not None this must be False.

    Attributes
    ----------
    targets : list of strs and/or lists
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
    isObjectQuery : bool or dict
        When archiveSearch is initialized with the `targets` argument then this
        is a dictionary of booleans that indicate whether each queried target
        is a source name (True) or a region (False). Targets are the keys and
        the boolean flags are the values. When archiveSearch is initialized
        with the `allSky` argument then this is False.
    invalidName : list of strs
        List of target strings specifying targets that did not return any
        results from the ALMA archive query.
    """

    def __init__(self, targets=None, allSky=False):
        if ((allSky and targets != None) or (not allSky and targets == None)):
            msg = 'Only one of either "targets" or "allSky" must be specified.'
            raise ValueError(msg)

        if allSky:
            self.isObjectQuery = False
        else:
            self.isObjectQuery = dict()

        self.targets = dict()
        if targets != None:
            for i in range(len(targets)):
                self.addTarget(targets[i])
        else:
            self.targets['All sky'] = 'All sky'

        self.invalidName = list()

        self.queryResults = dict()
        self.queryResultsNoNED = dict()
        self.queryResultsNoNEDz = dict()

        self.uniqueBands = dict()

    def runQueries(self, public=False, science=False, **kwargs):
        """Run requested queries.

        Parameters
        ----------
        public : bool
            Return only publicly available datasets?
        science : bool
            Return only data marked as "science" in the archive?
        kwargs : dict
            Keywords that are accepted by the ALMA archive system. You can look
            these up by examining the forms at http://almascience.org/aq.
            Passed to `astroquery.alma.Alma.query`. If archiveSearch was
            initialized with the `targets` argument then "source_name_resolver"
            and "ra_dec" cannot be used here.

        Also does some work on the result tables to put data into more useful
        forms. This includes:

          * converting the 'Release' and 'Observation' data columns from
            strings to np.datetime64 objects
        """
        if self.isObjectQuery == False:
            payload = dict()
            self.queryResults['All sky'] = Alma.query(payload,
                                                      public=public,
                                                      science=science,
                                                      **kwargs)
        else:
            if 'source_name_resolver' in kwargs:
                msg = '"source_name_resolver" cannot be used when ' \
                      + 'archiveSearch is initialized with the "targets" ' \
                      + 'argument.'
                raise ValueError(msg)

            if 'ra_dec' in kwargs:
                msg = '"ra_dec" cannot be used when archiveSearch is ' \
                      + 'initialized with the "targets" argument.'
                raise ValueError(msg)

            for target in self.targets:
                payload = dict()
                if self.isObjectQuery[target] == True:
                    payload['source_name_resolver'] = target
                else:
                    tarTmp = self.targets[target]
                    cstr = tarTmp[0].fk5.to_string(style='hmsdms', sep=':')
                    payload['ra_dec'] = '{:}, {:}'.format(cstr, tarTmp[1].deg)
                try: 
                    self.queryResults[target] = Alma.query(payload,
                                                           public=public,
                                                           science=science,
                                                           **kwargs)
                except ValueError:
                    self.invalidName.append(target)
                    print('Invalid name "{:}"'.format(target))
            for key in self.invalidName:
                self.targets.pop(key)

        self._convertDateColumnsToDatetime()

    def runQueriesWithLines(self, restFreqs, redshiftRange=(0, 1000),
                            lineNames=[], public=False, science=False,
                            **kwargs):
        """Run queries for spectral lines.

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
            Keywords that are accepted by the ALMA archive system. You can look
            these up by examining the forms at http://almascience.org/aq.
            Passed to `astroquery.alma.Alma.query`. "frequency" cannot be
            specified here since it is used to limit the query to frequencies
            that could contain the lines in the specified redshift range. If
            archiveSearch was initialized with the `targets` argument then
            "source_name_resolver" and "ra_dec" also cannot be used here.
        """
        if 'frequency' in kwargs:
            msg = '"frequency" cannot be passed to runQueriesWithLines'
            raise ValueError(msg)

        lineNames = np.array(lineNames)
        if (len(lineNames) != len(restFreqs) and len(lineNames) != 0):
            msg = 'length mismatch between '
            msg += '"restFreqs" ({:})'.format(len(restFreqs))
            msg += ' and "lineNames" ({:})'.format(len(lineNames))
            raise ValueError(msg)

        # generate line names if not set
        if len(lineNames) == 0:
            lineNames = ['Line{:}'.format(i) for i in range(len(restFreqs))]

        # sort frequencies and line names in assending frequency order
        restFreqs = np.array(restFreqs)
        lineNames = np.array(lineNames)

        restFreqOrderIdx = restFreqs.argsort()
        restFreqs = restFreqs[restFreqOrderIdx]
        lineNames = lineNames[restFreqOrderIdx]

        # sort redshift range
        redshiftRange = np.array(redshiftRange)
        redshiftRange.sort()

        # define frequency range from lines and redshifts
        lowFreq = self._observedFreq(np.sort(restFreqs)[0], redshiftRange[1])
        highFreq = self._observedFreq(np.sort(restFreqs)[-1], redshiftRange[0])
        freqLimits = '{:} .. {:}'.format(lowFreq, highFreq)

        self.runQueries(public=public, science=science, frequency=freqLimits,
                        **kwargs)

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
                
                if len(nedResult) > 0:
                    nedResult = vstack(nedResult)
                else:
                    msg = 'No NED results returned. nedResult = {:}'.format(nedResult)
                    raise ValueError(msg)

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
        target : str or list
            Target to query the ALMA archive for. Can be either a string
            indicating a source name (e.g. 'M87') or a list indicating a
            region to search consisting of (coordinates, radius). The
            coordinates element can be either a string or an
            astropy.coordinates object and the radius element can be either a
            string or an astropy.units.Quantity object.
        """
        targetType = type(target)

        if targetType == str:
            self.targets[target] = target
            self.isObjectQuery[target] = True
        elif targetType == list:
            if type(target[0]) != SkyCoord:
                target[0] = commons.parse_coordinates(target[0])
            if type(target[1]) != Angle:
                target[1] = Angle(target[1])

            targetStr = 'coord=({:} {:}) radius={:}deg'
            targetStr = targetStr.format(target[0].ra,
                                         target[0].dec,
                                         target[1].deg)

            self.targets[targetStr] = target
            self.isObjectQuery[targetStr] = False
        else:
            msg = 'Cannot work with targets of type {:}'.format(targetType)
            raise TypeError(msg)

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
        for tar in self.targets:
            targetFreqRanges = list()
            freqUnit = self.queryResults[tar]['Frequency support'].unit
            for i in range(len(self.queryResults[tar])):
                freqStr = self.queryResults[tar]['Frequency support'][i]
                freqStr = freqStr.split('U')
                rowFreqRanges = list()
                for j in range(len(freqStr)):
                    freqRange = freqStr[j].split(',')
                    # in a few cases (solar observations?) there is only one frequency. This handles that rather roughly.
                    if '[' in freqRange[0]:
                        freqRange = freqRange[0].strip(' [')
                        freqRange = freqRange.split('..')
                        freqRange[1] = freqRange[1].strip(string.ascii_letters)
                        freqRange = np.array(freqRange, dtype='float')
                        rowFreqRanges.append(freqRange)
                    else:
                        rowFreqRanges.append('No frequency range')
                
                targetFreqRanges.append(rowFreqRanges)
                    
            self.queryResults[tar]['Frequency ranges'] = targetFreqRanges
            self.queryResults[tar]['Frequency ranges'].unit = freqUnit

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
