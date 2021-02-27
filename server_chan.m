function x = server_chan(text, desp)
% desp supprot markdown; Chekck https://github.github.com/gfm/ for
% supproted syntax

    if nargin == 1
        desp = "";
    end
    SCKEY = "SCU150562Tdb67f760e198b961f10dfc82d506b7e36000d9e03d845";
    
    if strlength(text)>256
        desp =  "<h2>" + text +"</h2>" +"      " +" <p>"+ desp + "</p>";
        text = "Title is tooooooo long";
        disp(text)
    end
    
    switch nargin
        case 2
            SCWriteURL = "http://sc.ftqq.com/"+SCKEY+".send" + "?text="+text+"&desp="+desp;
            response = webwrite(SCWriteURL,'');
            re = split(response,",");
            x=~str2double(re{1}(end));
        case 1
            SCWriteURL = "http://sc.ftqq.com/"+SCKEY+".send" + "?text="+text+"&desp="+desp;
            response = webwrite(SCWriteURL,'');
            re = split(response,",");
            x=~str2double(re{1}(end));
         
        otherwise
            disp('Input at lease one string');
            x=false;
    end
end


