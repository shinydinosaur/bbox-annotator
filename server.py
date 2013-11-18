import os
import json
import web
import sys
import math
import random

urls = (
  '/', 'index'
)

render = web.template.render('templates/')
db = web.database(dbn='postgres', user='picasso', pw='picasso', db='picasso_annotations')
app = web.application(urls, globals())

if web.config.get('_session') is None: 
  session = web.session.Session(app, web.session.DiskStore('sessions')) 
  web.config._session = session 
else: 
  session = web.config._session 

NUM_USERS = 30
IMAGE_DIR = "static/data/img/"

class index:
  def user_images(self, uid):
    ''' returns the list of images for a user to annotate
     assuming the uid is a 1-indexed integer '''

    all_images = [IMAGE_DIR + i for i in os.listdir(IMAGE_DIR)]
    window_sz = int(math.ceil(len(all_images)/float(NUM_USERS)))
    start = (uid-1)*window_sz
    end = start+int(math.ceil(len(all_images)/3.0))
    images = all_images[start:end]
    if end > len(all_images):
        images.extend(all_images[0:end-len(all_images)])
    return images    

  def GET(self):
    if not session.get('uid'):

      # new session: show intro
      session.uid = db.insert('users')
      return render.intro()

    uid = session.uid
    images = self.user_images(uid)

    # figure out which images this user already annotated
    query_context = {"userid": uid}
    done_images = [result['imgid'] for result in
                   db.select('tasks', query_context,  what='imgid', where="userid = $userid")]
    remaining_images = list(set(images)-set(done_images))
    if not remaining_images:
      return render.done()

    # pick at random one of the images that was not yet annotated by this user
    current_image = random.choice(remaining_images)
    progress = 100 * float(len(done_images)) / len(images)
    return render.picasso_annotator(current_image, progress)

  def POST(self):
    i = web.input()
    uid = session.get('uid')
    if not uid:
      return "error: no session id"

    # task: user/image
    taskid = db.insert('tasks', userid=uid, imgid=i.imgid, 
                       time=float(i.time), difficulty=int(i.bucket))

    # annotation: user/image/object
    entries = json.loads(i.entries) 
    for box in entries:
       n = db.insert('annotations', userid=uid, imgid=i.imgid, taskid=taskid,
                     bbox_left=box["left"], bbox_top=box["top"], bbox_width=box["width"], bbox_height=box["height"]),
    raise web.seeother('/')

if __name__ == "__main__":
  if len(sys.argv) == 1:
    sys.argv.append("8005") # add the default port to run on
  app.run()
