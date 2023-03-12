import xmltodict

# enwrap file with <root>  </root>
# is performed by the function: replace & with &amp;
def read_data(fn):
    with open(fn, 'r', encoding="UTF-8") as file_xml: 
        file_content = file_xml.read().replace("&","&amp;")
        doc = xmltodict.parse(file_content)
    data=[]
    keywords = ["title","authors","body","copyright","isbn"]
    labels = []
    for b1,book in enumerate(doc["root"]["book"]):
        data.append(book)
        if "categories" not in book:
            continue
        label_t=[]
        if len(book["categories"]["category"])==1:
            topics=book["categories"]["category"]
            if type(topics["topic"])!=type(list()):
                label_t.append(topics["topic"]["@d"]+"_"+topics["topic"]["#text"].replace("&amp;","&"))
            else:
                for topic in topics["topic"]:
                    label_t.append(topic["@d"]+"_"+topic["#text"].replace("&amp;","&"))
        else:
                                
            for topics in book["categories"]["category"]:
                if type(topics["topic"])==type(list()):
                    d_topic=topics["topic"]
                    for topic in d_topic:
                        label_t.append(topic["@d"]+"_"+topic["#text"].replace("&amp;","&"))
                else:
                    label_t.append(topics["topic"]["@d"]+"_"+topics["topic"]["#text"].replace("&amp;","&"))
       
                #label_t.append([tk.values()[0]+"_"+tk.values()[1] for tk in topics])
        labels.append(label_t)
    return data,labels

def read_hierarchy(fn):
    parents_raw={}
    with open(fn, 'r') as file_csv:
        for l1 in file_csv:
            ws=l1.split("\t")
            parents_raw[ws[1].strip()]=ws[0]
    levels={}
    roots=[]
    for item in parents_raw.values():
        if item not in parents_raw.keys():
            levels[item]=0
            roots.append(item)
    for i in range(1,3):
        mkeys=list(levels.keys())
        for item in mkeys:
            for child in parents_raw.keys():
                if parents_raw[child]==item:
                    
                    if child not in levels:
                        levels[child]=levels[item]+1

    hierarchy={"ROOT":[]}
    for tk,tv in parents_raw.items():
        par=str(levels[tv])+"_"+tv
        if par in hierarchy:
            hierarchy[par].append(str(levels[tk])+"_"+tk)
        else:
            hierarchy[par]=[str(levels[tk])+"_"+tk]
    for par in hierarchy:
        if len([tk for tk in hierarchy.values() if par in tk ])==0:
            if par=="ROOT":
                continue
            hierarchy["ROOT"].append(par)
    return hierarchy,levels

if __name__=="__main__":
    data,labels=read_data("blurbs_train.xml")
    hierarchy=read_hierarchy("hierarchy.txt")
    
