# use keywords and negation to extract information and output the polarity and where the evidence coming from
import pandas as pd
import numpy as np
import glob
import re
import spacy
import nltk
import copy
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
punc = list(punc)
def generate_keyword():
    keyword_list = {"Opioid terms":"fentanyl, heroin, hydromorphone, dilaudid,  oxymorphone, opanum, opana, methadone, oxycodone, oxycotin, roxicodone, percocet, morphine, hydrocodone, vicodin, lortab, codeine, meperidine, demerol, tramadol, ultram, meloxicam, kratom, carfentanil, buprenorphine, meperidine, narcotic, dihydrocodeine, levorphanol, naloxone, naltrexone, pentazocine, suboxone, subutex, tapentadol, vivitrol, opiate, opioid, opium, opioid",
    "Use disorder terms":"abuse, abuses, abused, abusive, abusing, addict, addicts, addicting, addicted, addiction, dependence, dependant, dependance, dependency, dependency, misuse, misuses, misused, misusing, mis using, mis-using, overdose, overdoses, overdoes, over dose, over dosed,od, over use, over used, overuse, use disorder, use-disorder, inject, injected, injects, injection, injecting, ivda, intravenous drug abuse, iv drug use, intravenous drug user, iv drug user, ivdu, intravenous drug abuse, iv drug abuse, iv drug abuse, iv drug abuser, withdrawal, withdraw, withdrew, withdrawling",
    "Negation terms":"absence, absent, deny, denies, denied, denying, do not, don’t, donnot, exclude, excluded, excludes, excluding, lack, lacked, lacks, lacking, negative, negation, never, no, no evidence, no history, no hx, no sign, no signs, not observed, not present, without, without evidence, suspect, suspected",
    "Specialized terms":"denies alcohol/illicits/tobac, denies drug, denies drug abuse, denies etoh, illicit drugs, denies history of alcohol, tobacco or illegal drug use, denies illicit drug use, denies illict or iv drug use, denies iv drug use, denies smoking, alcohol, illicit drug use, denies tobacco or illicit drug use, denies tobacco, alcohol, drug use, denies tobacco/etoh/illicit drug use, drug abuse:denies, drugs: denies, illicit drug use: denies, illicits:denies, ivda/intranasal: denies, negative for current tobacco, alcohol, or recreational drug use, recreational drugs: denies",
    "Specific clinic":"first bridge clinic, the bridge"}

    def transform_lower_list(mylist):
        return map(lambda x:x.lower(),mylist)
    for k, v in keyword_list.items():
        keyword_list[k] = list(map(lambda x: x.lower() ,keyword_list[k].split(', ')))
        print(keyword_list[k])
    #listkeys = sum([*keyword_list.values()],[])
    return keyword_list
def read_txt_csv():
    csvdic ={"VisitID":[],"DocumentName":[],"DocumentDate":[],"Label":[],"Text":[]}
    for f in glob.glob("./*.txt"):
        print(f)
        annf = f.split(".txt")[0]+".ann"
        with open (annf, 'r') as annff:
            annlist = annff.read().split("\n")
        with open (f,'r') as contentf:
            contentstr = contentf.read()
            dischargedate = re.findall("Discharge Date.*?\]", contentstr)
            admissiondate = re.findall("Admission Date.*?\]", contentstr)
            if len(dischargedate):
                csvdic['VisitID'].append("/".join(re.findall("[\d.]*\d+",dischargedate[0])))
            else:
                csvdic['VisitID'].append("/".join(re.findall("[\d.]*\d+",admissiondate[0])))

            csvdic['DocumentName'].append(f.split(".txt")[0])
            csvdic['DocumentDate'].append("/".join(re.findall("[\d.]*\d+",admissiondate[0])))
            csvdic['Label'].append(" ".join(annlist))
            csvdic['Text'].append(contentstr)
    csvdf = pd.DataFrame(csvdic)
    csvdf.to_csv("summary_test.csv")
def chunk(pos,v):
    # have more than one sentence
    if v.count(".")>1:
        lastdot = len(v[:pos])-1-v[:pos][::-1].find('.')
        nextdot = v[pos:].find('.')+ pos
    else:
        lastdot = 0
        nextdot = len(v)
    return v[lastdot:nextdot]
def lemmatize(note, nlp):
    doc = nlp(note)
    lemNote = [wd.lemma_ for wd in doc]
    return " ".join(lemNote)
def neg_model():
    import scispacy
    from negspacy.negation import Negex
    nlp = spacy.load("en_ner_bc5cdr_md", disable = ['parser'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    negex = Negex(nlp)
    nlp.add_pipe(negex)
    return nlp
def checkneg(t):
    results = []
    nlp = neg_model()
    doc = nlp(t)
    for e in doc.ents:
        rs = str(e._.negex)
        if rs == "True":
            results.append(e.text)
    if len(results):
        return True
    else:
        return False

def rule5(words,keyword_list,judge,documentname,deid,section):
     
    def spaceless3(s,indextup): # A function to check whether second word and first one has no more than 3 words
        first = s[0]
        second = s[1]
        sentence = " ".join(words)[indextup[0]:indextup[1]]
        firstindex = sentence.index(first)
        secondindex = sentence.index(second)
        partsen12 = sentence[firstindex+len(first)+1:secondindex-1]
        punk12 =  set(punc)&set(partsen12.split())
        if len(s)==3:
            third = s[2]
            thirdindex = sentence.index(third)
            partsen23 = sentence[secondindex+len(second)+1:thirdindex-1]
            punk23 =  set(punc)&set(list(partsen23))
       
        if len(nltk.word_tokenize(partsen12))<=3 and len(punk12)==0:
            if len(s)==3:
                if len(punk23)==0:
                    return True
                else:
                    return False
            else:
                return True
            return True
        else:
            return False

    def spaceless3_23(s,indextup): # A function to check whether second word and first one has no more than 3 words
        first = s[0]
        second = s[1]
        third = s[2]
        sentence = " ".join(words)[indextup[0]:indextup[1]]
        firstindex = sentence.index(first)
        secondindex = sentence.index(second)
        thirdindex = sentence.index(third)
        partsen12 = sentence[firstindex+len(first)+1:secondindex-1]
        punk12 =  set(punc)&set(partsen12.split())
        partsen23 = sentence[secondindex+len(second)+1:thirdindex-1]
        punk23 =  set(punc)&set(list(partsen23))
        if len(nltk.word_tokenize(partsen23))<=3 and len(punk12)==0 and len(punk23)==0:
            return True
        else:
            return False


    def checkis(s):
        second = s[1]
        third = s[2]
        sentence = " ".join(words)
        secondindex = sentence.index(second)
        thirdindex = sentence.index(third)
        sentence = " ".join(words)
        partsen = sentence[secondindex+len(second)+1:thirdindex-1]
        if " is " in partsen:
            return True
        else:
            return False

    # build the regular expression for all dictionaries
    rop = r'(\b%s)'% r'|\b'.join(keyword_list["Opioid terms"])
    rdisorder = r'(\b%s)'% r'|\b'.join(keyword_list["Use disorder terms"])
    rspecial = r'(\b%s)'% r'|\b'.join(keyword_list["Specialized terms"])
    rneg = r'(\b%s)'% r'|\b'.join(keyword_list["Negation terms"])
    rclinic = r'(\b%s)'% r'|\b'.join(keyword_list["Specific clinic"])

    # if has neg words go to rule 3 and rule 4
    if set(words) & set(keyword_list["Negation terms"]):
        # extract negwords
        neg_words = re.findall(re.compile("%s" %(rneg))," ".join(words))
        negindex = [(m.start(0), m.end(0)) for m in re.finditer(re.compile("%s" %(rneg))," ".join(words))]
        
        # Rule 3 
        if set(words) & set(keyword_list["Opioid terms"]) and set(words) & set(keyword_list["Use disorder terms"]):
            opterms = re.findall(re.compile("%s" %(rop))," ".join(words)) # extract Opioid terms
            opindex = [(m.start(0), m.end(0)) for m in re.finditer(re.compile("%s" %(rop))," ".join(words))]
            disorderterms = re.findall(re.compile("%s" %(rdisorder))," ".join(words)) # extract disorderterms
            
            ##############3##########################
            reg3 = re.compile("%s+.*?%s+.*?%s" %(rneg,rop,rdisorder))
            reg3index = [(m.start(0), m.end(0)) for m in re.finditer(reg3," ".join(words))]
            candidate_phrase03 = []
            copywords = copy.deepcopy(words)
            copywords = " ".join(copywords)
            strwords = " ".join(words)
            # get all possible phrase by looping all possible neg words
            for ind3, x3 in enumerate(neg_words):
                tempphrase = re.findall(reg3,copywords)
                candidate_phrase03 += tempphrase
                copywords = strwords[negindex[ind3][1]+1:]
            # filter out those phrase if first word and second word has more than 3 valid words
            candidate_phrase3 = list(filter(lambda x:spaceless3(x[0],x[1]),zip(candidate_phrase03,reg3index)))
            # save those extracted phrases 
            if candidate_phrase3:
                for c3 in candidate_phrase3:
                    c3 = c3[0]
                    judge['DeID'].append(deid)
                    judge['DocumentName'].append(documentname)
                    judge['Sections'].append(section)
                    judge['Keyterms'].append(" ".join(c3))
                    judge['Rule'].append("Rule 3")
                
            
            ##############Rule 4##########################
            reg4 = re.compile("%s+.*?%s+.*?%s" %(rop,rdisorder,rneg))
            reg4index = [(m.start(0), m.end(0)) for m in re.finditer(reg4," ".join(words))]
            candidate_phrase04 = []
            copywords = copy.deepcopy(words)
            copywords = " ".join(copywords)
            strwords = " ".join(words)
            # get all possible phrase by looping all possible neg words
            for ind4,x4 in enumerate(opterms):
                tempphrase = re.findall(reg4,copywords)
                candidate_phrase04 += tempphrase
                copywords = strwords[opindex[ind4][1]+1:]
            # filter out those phrase if second word and third word has more than 3 valid words
            candidate_phrase4 = list(filter(lambda x:spaceless3_23(x[0],x[1]),zip(candidate_phrase04,reg4index)))
        
            if candidate_phrase4: 
                for c4 in candidate_phrase4:
                    c4 = c4[0]
                    judge['DeID'].append(deid)
                    judge['DocumentName'].append(documentname)
                    judge['Sections'].append(section)
                    judge['Keyterms'].append(" ".join(c4))
                    judge['Rule'].append("Rule 4")
    else:
        ### rule 1, 2, 6
        if set(words) & set(keyword_list["Opioid terms"]) and set(words) & set(keyword_list["Use disorder terms"]):   
            #opterms = set(words) & set(keyword_list["Opioid terms"])
            #disorderterms = set(words) & set(keyword_list["Use disorder terms"])
            opterms = re.findall(re.compile("%s" %(rop))," ".join(words))
            opindex = [(m.start(0), m.end(0)) for m in re.finditer(re.compile("%s" %(rop))," ".join(words))]
            disorderterms = re.findall(re.compile("%s" %(rdisorder))," ".join(words))
            disorderindex = [(m.start(0), m.end(0)) for m in re.finditer(re.compile("%s" %(rdisorder))," ".join(words))]
            ##############1##########################
            reg1 = re.compile("%s+.*?%s" %(rop,rdisorder))
            reg1index = [(m.start(0), m.end(0)) for m in re.finditer(reg1," ".join(words))]
            candidate_phrase01 = []
            copywords = copy.deepcopy(words)
            copywords = " ".join(copywords)
            strwords = " ".join(words)
            # get all possible phrase by looping all possible Opioid terms
            for ind1,x1 in enumerate(opterms):
                tempphrase = re.findall(reg1,copywords)
                candidate_phrase01 += tempphrase
                copywords = strwords[opindex[ind1][1]+1:]
            # filter out those phrase if first word and second word has more than 3 valid words
            candidate_phrase1 = list(filter(lambda x:spaceless3(x[0],x[1]),zip(candidate_phrase01,reg1index)))
            if candidate_phrase1: 
                for c1 in candidate_phrase1:
                    c1 = c1[0]
                    judge['DeID'].append(deid)
                    judge['DocumentName'].append(documentname)
                    judge['Sections'].append(section)
                    judge['Keyterms'].append(" ".join(c1))
                    judge['Rule'].append("Rule 1")

            ##############2##########################
            reg2 = re.compile("%s+.*?%s" %(rdisorder,rop))
            reg2index = [(m.start(0), m.end(0)) for m in re.finditer(reg2," ".join(words))]
            candidate_phrase02 = []
            copywords = copy.deepcopy(words)
            copywords = " ".join(copywords)
            strwords = " ".join(words)
            # get all possible phrase by looping all possible disorder terms
            for ind2, x2 in enumerate(disorderterms):
                tempphrase = re.findall(reg2,copywords)
                candidate_phrase02 += tempphrase
                copywords = strwords[disorderindex[ind2][1]+1:]
            # filter out those phrase if first word and second word has more than 3 valid words
            candidate_phrase2 = list(filter(lambda x:spaceless3(x[0],x[1]),zip(candidate_phrase02,reg2index)))
            if candidate_phrase2:
                for c2 in candidate_phrase2:
                    c2 = c2[0]
                    judge['DeID'].append(deid)
                    judge['DocumentName'].append(documentname)
                    judge['Sections'].append(section)
                    judge['Keyterms'].append(" ".join(c2))
                    judge['Rule'].append("Rule 2")

        ##################Rule 6########################
        regclinic = re.compile("%s" %(rclinic))
        clinic_term = re.findall(regclinic," ".join(words)) # extract specific clinic term
        if clinic_term:
            for cclinic in clinic_term:
                judge['DeID'].append(deid)
                judge['DocumentName'].append(documentname)
                judge['Sections'].append(section)
                judge['Keyterms'].append(" ".join(cclinic))
                judge['Rule'].append("Rule 6")

    
    ##################5########################
#    regspecial = re.compile("%s" %(rspecial))
#    if set(words) & set(keyword_list["Specialized terms"]):
#        #special_term = set(words) & set(keyword_list["Specialized terms"])
#        #special_term = " ".join(list(special_term))
#        special_term = re.findall(regspecial," ".join(words))
#        judge['DeID'].append(deid)
#        judge['DocumentName'].append(documentname)
#        judge['Sections'].append(section)
#        judge['Keyterms'].append(", ".join(special_term))
#        judge['Rule'].append("Rule 5")




def checkkeyword(keyword_list,eachseg,judge,documentname,deid):
    nlp0 = spacy.load("en_core_sci_sm")
    # loop each segment
    for k, v in eachseg.items(): # k is parsed section
        lemv = lemmatize(v, nlp0)
        lemv = lemv.lower() # lowercase current segment
        newv = lemv.split()
        checktwo = keyword_list["Opioid terms"]
        rclinic = r'(\b%s)'% r'|\b'.join(keyword_list["Specific clinic"])
        regclinic = re.compile("%s" %(rclinic))
        clinic_term = re.findall(regclinic,lemv)
        
        # check current segement has overlapping with Opioid terms or clinic terms or not
        if set(checktwo) & set(newv) or len(clinic_term)>0:
            # toknize segement into sentence
            sentences = nltk.tokenize.sent_tokenize(lemv)
            # loop each sentence
            for s in sentences:
                words = s.split()
                # check current sentence has overlapping with Opioid terms or clinic terms or not
                if set(checktwo) & set(words) or len(clinic_term)>0:
                    # pull current sentence for allrules
                    rule5(words,keyword_list,judge,documentname,deid,k)
def read_csv_parse(keyword_list):
    # read file
    df = pd.read_csv("testingcases.csv")
    writeout =1
    # build output file column
    judge ={"DeID":[],"DocumentName":[],"Sections":[],"Keyterms":[],"Rule":[]}

    # loop each row of the csv
    for index, row in df.iterrows():
        #judge["DocumentName"].append(row["DocumentName"])

        # prepare segement
        text = row['ValueText']
        text = text.replace("\n",". ")
        text = text.replace(";",".")
        segment = re.findall(r":", text)
        secindex = [(m.start(0), m.end(0)) for m in re.finditer(r":",text)]
        textcp = ('.'+text + '.')[:-1]
        eachseg ={}
        report = []
        lastkey = "basic_info"
        lastcolon = 0
        if len(segment):
            for sindex in range(len(secindex)): 
                precolon = textcp[:secindex[sindex][0]+1]
                predot = re.finditer(r"\.",precolon)
                predot = list(predot)[-1].start(0)
                if predot <lastcolon:
                    predot = re.finditer(r"(;|,|\n|\?)",precolon)
                    predot = list(predot)[-1].start(0)
                    if predot <lastcolon:
                        predot = re.finditer(r"\s",precolon)
                        predot = list(predot)[-1].start(0)

                sec = precolon[predot+1:secindex[sindex][1]]
                sec = sec.strip()
                pretxt = textcp[lastcolon+1:predot+1]
                #posttxt = textcp[secindex[sindex][1]+1:]
                eachseg[lastkey] = pretxt.strip("\n")
                lastkey = sec 
                lastcolon = secindex[sindex][1]
            eachseg[lastkey] = textcp[lastcolon+1:]

        else:
            eachseg["wholeinfo"] = textcp.strip("\n")
        # process the segment
        checkkeyword(keyword_list, eachseg,judge,row["DocumentName"],row['DeID'])
    # output csv file
    if writeout:
        judgedf = pd.DataFrame(judge)
        judgedf.to_csv("judge_res_new.csv",index=False)
def pipeline():
    # get keywords to dictionary
    keyword_list = generate_keyword()
    # read csv file
    read_csv_parse(keyword_list)
pipeline()
