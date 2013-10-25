$(document).ready(function() {
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
  })
});
