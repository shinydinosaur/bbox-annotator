function visualize_annotations(annotations_file, img_path)
% prepare different colors
colors = colormap('lines');

T = readtable(annotations_file);
% get the names of all images
img_names = unique(T.imgid);
% go over each image
for i = 1:length(img_names)
    path = img_names(i);
    path = path{1};
    parts = strsplit(path,'/');
    name = parts{4};
    % get all annotations for a single image
    single_img_annotations = T(strcmp(T.imgid,path),:);
    % visualize the annotations for this image
    % read in the image from the pile of images
    I = imread([img_path name]);
    % go over all user ids that annotated this image
    users = unique(single_img_annotations.userid);
    for u = 1:length(users)
        % get the single user's annotations for this image
        u_ann = single_img_annotations(single_img_annotations.userid == users(u),:);
        % bboxes an M-by-4 matrix defining M bounding boxes
        % containing the detected objects each with [x,y,w,h] format.
        bboxes = [u_ann.bbox_left, u_ann.bbox_top, u_ann.bbox_width, u_ann.bbox_height]; % bug when more than one annotations per user?
        I = insertObjectAnnotation(I, 'rectangle', bboxes, u_ann.userid, 'Color', 255*colors(u,:));
    end
    figure
    imshow(I);
    title(name);
end
end