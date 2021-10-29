## Script written by Jingfeng Yang
'''import PyPDF2 
 

if __name__ == "__main__": 
    # creating a pdf file object 
    pdfFileObj = open('jingfeng.pdf', 'rb') 
    
    # creating a pdf reader object 
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    
    # printing number of pages in pdf file 
    print(pdfReader.numPages) 
    
    # creating a page object 
    pageObj = pdfReader.getPage(0) 
    
    # extracting text from page 
    print(pageObj.extractText()) 
    
    # closing the pdf file object 
    pdfFileObj.close()  '''

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from nltk.tokenize import sent_tokenize

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams() # set parameters for analysis
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()

    text = text.replace('-\n', '').replace('\n', ' ')
    sents = sent_tokenize(text)
    for sent in sents:
        print(sent)   
    return sents

if __name__ == "__main__": 
    convert_pdf_to_txt('roberta.pdf')