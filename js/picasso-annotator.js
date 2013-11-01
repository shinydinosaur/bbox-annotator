$(document).ready(function() {
  // grab some timing info
  var t0 = window.performance.now();

  // Initialize the bounding-box annotator.
  var annotator = new BBoxAnnotator({
    url: "/Users/shiry/Documents/Projects/Art_Vision/Picasso/People/A-Young-Faun-Playing-a-Serenade-to-a-Young-Girl-1938.JPG",
    input_method: 'fixed',    // Can be one of ['text', 'select', 'fixed']
    labels: "object",
    onchange: function(entries) {
      // Input the text area on change. Use "hidden" input tag unless debugging.
      // <input id="annotation_data" name="annotation_data" type="hidden" />
      // $("#annotation_data").val(JSON.stringify(entries))
      $("#annotation_data").text(JSON.stringify(entries, null, "  "));
    }
  });
  
  // Initialize the reset button.
  $("#reset_button").click(function(e) {
    annotator.clear_all();
  });

  // When a button is clicked in the is this human? buckets question.
  $("#buckets").button(function() {
  	$("#bucket").val($(this).text());
  });

  $("#submit").click(function() { 
  	// how are we passing data?
  	// entries, bucket, time
	var t1 = window.performance.now();
	var time = t1-t0; // in milliseconds
  });
});
