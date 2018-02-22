import web, cgi, sys
import settings
from dog_breed_predictor import predict_img, face_detector, dog_detector

urls = (
    '^/dog-breed/new[/]?$', 'do_new',
    '^/dog-breed/predict[/]?$', 'do_predict',
    '^/dog-breed/error[/]?$', 'do_error'   
)
app = web.application(urls, globals())

render = web.template.render(settings.TEMPLATE_FOLDER)

class do_new:
    def GET(self):
        print('Gather info for NEW prediction')        
        return render.new_prediction(title = 'New dog breed prediction')

class do_predict:
    def POST(self):
        cgi.maxlen = settings.MAX_UP_FILE_SIZE

        input = web.input(dog_file = {}, dog_name = 'Firulais')
        if input.dog_file.file:
            try:
                dog_file_name = self.save_file(input.dog_file)

                print('New Peediction in progress: {}'.format(dog_file_name))

                if face_detector(dog_file_name):
                    dog_status = 0
                elif dog_detector(dog_file_name):
                    dog_status = 1
                else:
                    dog_status = -1
                
                dog_breed = predict_img(dog_file_name)

            except Exception as e:
                print('Unexpected error: {}'.format(str(e)))
                raise web.seeother('/dog-breed/error')
                
        return render.prediction(input.dog_name, dog_breed, dog_file_name, dog_status)
    
    def save_file(self, dog_file):
        filedir = settings.STATIC_FOLDER
        file_ext = dog_file.filename.split(".")[-1]
        filename = filedir + 'another_dog.' + file_ext
        fout = open(filename,'wb')
        fout.write(dog_file.file.read()) 
        fout.close() 
        return filename
        

class do_error:
    def GET(self):
        print('ERROR.............................................')
        return render.error_page(error_description = 'Internal Server Horror')
    
            
        
if __name__ == "__main__":
    app.run()