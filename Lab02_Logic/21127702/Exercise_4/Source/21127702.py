import time
import os
import itertools

def readFile(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    m = 1
    queryClauses = [lines[0].strip()]
    
    n = int(lines[m].strip())
    KBClauses = [line.strip() for line in lines[m+1:]]

    return queryClauses, KBClauses


def negate(w):
    if w[0] == "-":
        return w[1:]
    else:
        return "-" + w[0]

def negateQuery(query):
    listClauses = []

    for clauses in query:
        clause = clauses.split()
        for i in clause:
            if (i != 'OR'):
                convertedClauses = []
                convertedClauses.append(negate(i))
                listClauses.append(convertedClauses)
    return listClauses
    
def generateKB(KBClauses, queryClauses):
    listClauses = []
    for clauses in KBClauses:
        clause = clauses.split()
        convertedClauses = []
        for i in clause:
            if (i != 'OR'):
                convertedClauses.append(i)
        listClauses.append(convertedClauses)

    query = negateQuery(queryClauses)
    for clause in query:
        listClauses.append(clause)

    return listClauses

def isNegate(a1, a2):
    if a1 == "-" + a2 or a2 == "-" + a1:
        return True
    return False

def isEquivalent(clause):
    for i in range(len(clause)-1):
        for j in range(i + 1, len(clause)):
            if isNegate(clause[i], clause[j]) or clause[i] == clause[j]:
                return True
    return False

def resolventInClauses(resolvents, clauses):
    listResolvents = list(itertools.permutations(resolvents))
    for res in listResolvents:
        if list(res) in clauses:
            return True
    return False

def pl_resolve(Ci, Cj):
    for i in Ci:
        if negate(i) in Cj:
            tempCi = Ci.copy()
            tempCi.remove(i)
            tempCj = Cj.copy()
            tempCj.remove(negate(i))
            if isEquivalent(tempCi + tempCj) == False:
                if (tempCi + tempCj == []):
                    return True, ["{}"]
                return True, tempCi + tempCj
    return False, ['None']

def pl_resolution(KB, query):
    clauses = generateKB(KB, query)
    result = []
    while True:
        newResolvents = []
        for i in range(len(clauses) - 1):
            for j in range(i + 1, len(clauses)):
                init, resolvents = pl_resolve(clauses[i], clauses[j])
                if init == True and resolventInClauses(resolvents, clauses) == False and resolventInClauses(resolvents, newResolvents) == False:
                    newResolvents.append(resolvents)

        result.append(newResolvents)
        if not newResolvents: 
            return result, False
        else:
            if ["{}"] in newResolvents:
                return result, True
            else:
                for res in newResolvents:
                    if res not in clauses and not isEquivalent(res):
                        clauses.append(res)

def custom_sort(item):
    return item.lstrip('-')

def fileOut(result, init, folderOut, filePathOut):
    fileOut = open(os.path.join(folderOut, filePathOut), 'w')  
    for res in result:
        print(len(res), file=fileOut)
        for resovants in res:
            if resovants == []:
                print("0", file=fileOut)
            else:
                sortedResovants = sorted(resovants, key=custom_sort)
                print(' OR '.join(sortedResovants), file=fileOut)
    if(init == True):
        print("YES", file=fileOut)
    else:
        print("NO", file=fileOut)
    fileOut.close()

filePathIn = input("Input file: ")
filePathOut = input("Output file: ")
folderIn = "21127702\Exercise_4\Input"
folderOut = "21127702\Exercise_4\Output"
q, k = readFile(os.path.join(folderIn, filePathIn))
result, init = pl_resolution(k, q)

fileOut(result, init, folderOut, filePathOut)