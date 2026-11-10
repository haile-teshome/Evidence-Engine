-- Evidence Engine's face in the Dock.
--
-- The app window is a Chrome window, and a Chrome window carries Chrome's icon
-- and Chrome's name. A plain shell script cannot fix that: a script is not an
-- app, so it has no tile of its own and macOS never tells it the icon was
-- clicked. This applet IS an app. It owns the tile, it stays in the Dock for as
-- long as the review is open, and it is told every time the icon is clicked,
-- which is what makes clicking it do the obvious thing.
--
-- It holds no logic of its own. Starting, serving and stopping all stay in
-- launch.mjs; this only forwards three events to it: opened, clicked again,
-- and quit.

on appRoot()
	-- Contents/Resources/app, beside this applet inside the same bundle.
	set here to POSIX path of (path to me)
	return here & "Contents/Resources/app"
end appRoot

on launcher()
	set root to appRoot()
	set node to root & "/runtime/node/bin/node"
	-- A source checkout has no bundled runtime; fall back to a system Node.
	set runner to "if [ -x " & quoted form of node & " ]; then NODE=" & quoted form of node & "; else NODE=node; fi; "
	-- Every stream is redirected and the shell is told to exit, because
	-- `do shell script` waits for its shell AND for anything still holding the
	-- output pipe. Written as `A && B &` it was the whole list that went to the
	-- background, in a subshell that sat waiting on the launcher, so this call
	-- never returned, `on run` never finished, and every later click on the
	-- Dock icon queued behind it and did nothing at all.
	return runner & "cd " & quoted form of root & "; EE_DOCK_OWNER=1 nohup \"$NODE\" " & quoted form of (root & "/launch.mjs") & " >/dev/null 2>&1 </dev/null & exit 0"
end launcher

-- The URL the running launcher recorded for its window, or "" if none.
on appURL()
	try
		set lock to (POSIX path of (path to home folder)) & ".evidence-engine/launcher.pid"
		return do shell script "sed -n '2p' " & quoted form of lock & " 2>/dev/null"
	on error
		return ""
	end try
end appURL

on running_()
	try
		set lock to (POSIX path of (path to home folder)) & ".evidence-engine/launcher.pid"
		do shell script "kill -0 $(sed -n '1p' " & quoted form of lock & ") 2>/dev/null"
		return true
	on error
		return false
	end try
end running_

on run
	-- An applet keeps its properties between launches, written back into its
	-- own script. Left alone, the "I have seen the engine" flag came back
	-- already true, so the very first idle tick of the NEXT launch saw no lock
	-- yet, believed the engine had died, and quit before the app had started.
	set sawEngine to false
	set ticks to 0
	do shell script my launcher()
end run

-- Every later click on the Dock icon. The window may be open, or the reviewer
-- may have closed it and come back; either way they want it in front of them.
on reopen
	set url_ to my appURL()
	if url_ is not "" and my running_() then
		try
			do shell script "curl -s -m 3 -X POST " & quoted form of (url_ & "__ee/open-window") & " >/dev/null"
			return
		end try
	end if
	do shell script my launcher()
end reopen

-- Quitting from the Dock is how a reviewer stops the whole thing: the window,
-- the engine and the local models, rather than leaving them running unseen.
on quit
	try
		set url_ to my appURL()
		if url_ is not "" then
			do shell script "curl -s -m 5 -X POST " & quoted form of (url_ & "__ee/quit") & " >/dev/null"
		end if
	end try
	continue quit
end quit

-- If the engine stops on its own, the tile should not linger pretending.
--
-- It must not quit before the engine has ever been seen, though. The first
-- idle arrives within a moment of launch, well before the launcher has written
-- its lock, and checking only "is it running" made the applet quit on its own
-- first tick and take the Dock tile with it. So it waits to see the engine
-- once, and gives up only if it never appears.
property sawEngine : false
property ticks : 0

on idle
	set ticks to ticks + 1
	if my running_() then
		set sawEngine to true
	else if sawEngine then
		quit
	else if ticks > 24 then
		-- Six minutes and no engine: something is wrong, and a tile that does
		-- nothing is worse than no tile.
		quit
	end if
	return 15
end idle
