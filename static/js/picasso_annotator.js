$(document).ready(function() {

  function getURLParameter(name) {
    return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search)||[,""])[1].replace(/\+/g, '%20'))||null
    }

  // if we have an imgid in our url, ignore the random id we got from the backend
  url_imgid = getURLParameter('imgid');
  if (url_imgid) {
    $("#imgid").val(url_imgid);
  }

  // grab some timing info
  var t0 = window.performance.now();
  //var imgid =  "static/img/A-Young-Faun-Playing-a-Serenade-to-a-Young-Girl-1938.JPG";
  var imgid = $("#imgid").val();

  // throw the image id into the url to control refresh behavior
  history.replaceState({}, "", "/?imgid="+imgid);

  // Initialize the bounding-box annotator.
  var annotator = new BBoxAnnotator({
    url: imgid,
    input_method: 'fixed',    // Can be one of ['text', 'select', 'fixed']
    labels: "object",
    onchange: function(entries) {
      // Input the text area on change. Use "hidden" input tag unless debugging.
      // <input id="annotation_data" name="annotation_data" type="hidden" />
      // $("#annotation_data").val(JSON.stringify(entries))
      $("#annotation_data").val(JSON.stringify(entries, null, "  "));
    }
  });
  
  // Initialize the reset button.
  $("#reset_button").click(function(e) {
    annotator.clear_all();
  });

  $("#submit").click(function() { 
      // how are we passing data?
      // entries, bucket, time

      // do some validation
      // Must pick a bucket
      var err_msg = "";
      if ($("#buckets :checked").val() == undefined) {
	  err_msg = "Please provide an answer to the lifelike question.";
      } 

      // Must box at least one object
      var annotations = $("#annotation_data").val();
      if (annotations == "" || annotations == "[]") {
	  err_msg = "Please draw a box around at least one object.";
      } 

      if (err_msg != "") {
	  $("#validation_error").text(err_msg);
	  $("#validation_div").removeClass("hidden");
	  event.preventDefault();
	  return;
      }

      // set hidden forms and submit.
      var t1 = window.performance.now();
      $("#imgid").val(imgid);
      $("#time").val(t1-t0); // in milliseconds
      $("#myform").trigger("submit");
  });
});
