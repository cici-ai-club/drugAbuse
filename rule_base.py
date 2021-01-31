# use keywords and negation to extract information and output the polarity and where the evidence coming from
import pandas as pd
import numpy as np
import glob
import re
import spacy
def generate_keyword():
    keyword_list = {"Opioid terms":"Amphetamine, Buprenorphine, Codeine, Dihydrocodeine, Dilaudid, Fentanyl, Heroin, Hydrocodone, Hydromorphone, Levorphanol, Lortab, Meperidine, Methadone,Morphine, Naloxone, Naltrexone, Norcotic, Opana, Opiate, Opioid, Opium, Oxy, Oxycodone, Oxymorphone, Percocet, Pentazocine, Suboxone, Subutex, SUD, Tapentadol, Tramadol, Vicodin, Vivitrol","Problem use terms":"Addict, Addiction, Addicted, Opioid Abuse, Opioid Use, Dependent, Dependent, Dependence, dependency, misuse, overuse, OUD, use disorder, Substance, Abuse, Substance Use, Over dose, Overdosed, disorder","Methods terms":"Inject, Injection, IVDA, IVDU, Needle","Status term":"Found down, Respiratory Arrest, Withdrawal","Locations":"First Bridge, The Ridge, Jail, Prison"}
    def transform_lower_list(mylist):
        return map(lambda x:x.lower(),mylist)
    for k, v in keyword_list.items():
        keyword_list[k] = list(map(lambda x: x.lower() ,keyword_list[k].split(', ')))
        print(keyword_list[k])
    listkeys = sum([*keyword_list.values()],[])
    return listkeys
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

def checkkeyword(listkeys,eachseg):
    nlp0 = spacy.load("en_core_sci_sm")
    res ={}
    for k, v in eachseg.items(): # k is parsed section
        lemv = lemmatize(v, nlp0)
        lemv = lemv.lower()
        newv = lemv.split()
        if set(listkeys) & set(newv):
            posx =0
            negx =0
            for x in set(listkeys) & set(newv):
                if not checkneg(chunk(lemv.find(x),lemv)):
                    posx +=1
                else:
                    negx +=1
            if posx >negx: # if pos larger than neg, then we record as drug abuse, otherwise not.
                res[k]=[(v,x,posx)]

    if not len(res):
        return (0,[],{})
    else:
        return (len(res), res.keys(),res) 
def read_csv_parse(listkeys):
    df = pd.read_csv("summary_test.csv")
    writeout =1
    judge ={"Abuse":[],"Sections":[],"Evidence":[],"DocumentName":[]}
    for index, row in df.iterrows():
        judge["DocumentName"].append(row["DocumentName"])
        text = row['Text']
        segment = re.findall("(.*?):\n", text)
        textcp = (text + '.')[:-1]
        eachseg ={}
        report = []
        lastkey = "basic_info" 
        if len(segment):
            for sindex, sec in enumerate(segment):
                pretxt = textcp[:textcp.find(sec+":\n")]
                posttxt = textcp[(textcp.find(sec+":\n")+len(sec+":\n")):]
                eachseg[lastkey] = pretxt.strip("\n")
                lastkey = sec
                textcp = posttxt
        else:
            eachseg["wholeinfo"] = textcp.strip("\n")
        numofsec, sections, evidence = checkkeyword(listkeys, eachseg)
        if writeout:
            if numofsec:
                judge["Abuse"].append(numofsec>0)
                judge["Sections"].append(sections)
                judge["Evidence"].append(evidence)
            else:
                judge["Abuse"].append(numofsec>0)
                judge["Sections"].append(sections)
                judge["Evidence"].append(evidence)
            judgedf = pd.DataFrame(judge)
            judgedf.to_csv("judge_res.csv",index=False)
def pipeline():
    listkeys = generate_keyword()
    read_csv_parse(listkeys)
pipeline()
