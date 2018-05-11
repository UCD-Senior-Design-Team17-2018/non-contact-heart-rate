% in_file = "vid1.mp4";
% in_file = "vid2.mp4";
in_file = "android_face.mp4";
% Get basic file info
file_info = aviinfo(in_file);
% Initialise the waveform vector
waveform = zeros(1,file_info.NumFrames);
% Extract frame data
for a_frame=1:file_info.NumFrames
    disp(sprintf("Working on frame %d / %d\n", a_frame, file_info.NumFrames));
    frame_image = aviread(in_file, a_frame);
    % Get the averages (from a specific rectangle
    % R = mean(mean(frame_image(318:522,578:1140,1)));
    % G = mean(mean(frame_image(318:522,578:1140,2)));
    % B = mean(mean(frame_image(318:522,578:1140,3)));
    % Get the averags (full frame)
    R = mean(mean(frame_image(:,:,1)));
    G = mean(mean(frame_image(:,:,2)));
    B = mean(mean(frame_image(:,:,3)));
   
    waveform(a_frame) = (R+G+B)./3.0;
    fflush(stdout);
end;