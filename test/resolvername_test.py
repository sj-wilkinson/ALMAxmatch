'''
Mar 25th, 2015

Test several names to make sure it's the resolver name. 
'''

from astroquery.alma import Alma

############################################################
## One of the normal galaxies
target='IRAS F17207-0014 '
table=Alma.query_object(target,science=False,public=False)
print(len(table))
# there are 11 projects. 

'''
Input the target name into the online search field 'resolver name', it returns 11 projects as expected. 
Input the target name into the online search field 'alma name', it can't identify the name
'''
'''
coclusion: the name is not interpreted as alma name. 
'''
############################################################
## the galaxy that couldn't be found in the command.
target='IRAS F12112+0305 '
# table=Alma.query_object(target,science=False,public=False)
# report the error.

table=Alma.query_object(target,science=True,public=False)
# print(table)
# it will return 14 objects

'''
Input the target name into the search field 'resolver name', it returns 14 objects. Results matches the online results. 
'''
'''
conclusion: The problem is not due to the target name is not resolved. It's due to the parameter science=False. 
'''

############################################################
## CGCG objects
target='CGCG 142-034'
table=Alma.query_object(target,science=False,public=False)
print(table)
# can't be resolved in the command

target='CGCG 142-034 ' # add a white space string
table=Alma.query_object(target,science=False,public=False)
print(table)
# the name could be resolved but not information returned. 

'''
This name could also not be interpreted in the online search
'''

'''
conlusion: the while space seems to affect the result. suggesting to strip the while space. 
'''
