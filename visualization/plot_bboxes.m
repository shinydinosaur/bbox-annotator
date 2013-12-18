function plot_bboxes(I, bboxes)

% plot_bboxes(im, bboxes)
% Draw boxes on top of image.
% bboxes an M-by-4 matrix defining M bounding boxes
% containing the detected objects each with [x,y,w,h] format.

IPeople = insertObjectAnnotation(I, 'rectangle', bboxes, 'Person');
figure, imshow(IPeople), title('Detected people');