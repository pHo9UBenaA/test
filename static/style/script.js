function loading(){
    $("#loading").show();
    $("#img_file").hide();
    $("#content").hide();
    $("#message").hide();
    $("#result_img").hide();
}

$(function() {
    $('.file').on('change', function () {
        var file = $(this).prop('files')[0];
        $('.filename').text(file.name);
    });
});

