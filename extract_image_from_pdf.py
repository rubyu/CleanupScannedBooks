import sys
import os
import PyPDF2
from PIL import Image

# http://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit("extract_image_from_pdf.py `pdf_file` `output_directory`")
    
    def output_path(index, name, ext):
        return os.path.join(sys.argv[2], "{0}-{1}.{2}".format(index, name, ext))
    
    input = PyPDF2.PdfFileReader(open(sys.argv[1], "rb"))
    for i in xrange(0, input.getNumPages()):
        page = input.getPage(i)
        xObject = page['/Resources']['/XObject'].getObject()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()
                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    mode = "P"
                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    img.save(output_path(i, obj[1:], "png"))
                elif xObject[obj]['/Filter'] == '/DCTDecode':
                    img = open(output_path(i, obj[1:], "jpg"), "wb")
                    img.write(data)
                    img.close()
                elif xObject[obj]['/Filter'] == '/JPXDecode':
                    img = open(output_path(i, obj[1:], "jp2"), "wb")
                    img.write(data)
                    img.close()
