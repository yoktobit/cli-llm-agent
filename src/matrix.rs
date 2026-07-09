use std::time::Duration;

use matrix_sdk::{Client, Room, RoomState, config::SyncSettings, ruma::events::room::{member::StrippedRoomMemberEvent, message::{MessageType, OriginalSyncRoomMessageEvent, RoomMessageEventContent}}, sleep::sleep};

use crate::adk::call_ai;

// pub async fn on_room_message(event: OriginalSyncRoomMessageEvent, room: Room) {
//     if room.state() != RoomState::Joined {
//         return;
//     }
//     let MessageType::Text(text_content) = event.content.msgtype else {
//         return;
//     };

//     if text_content.body.contains("!party") {

//         let party_content = call_ai("Explain what a party is.".to_string()).await.unwrap();

//         let content = RoomMessageEventContent::text_plain(party_content);

//         println!("sending");

//         // send our message to the room we found the "!party" command in
//         room.send(content).await.unwrap();

//         println!("message sent");
//     }
// }

// pub async fn on_stripped_state_member(
//     room_member: StrippedRoomMemberEvent,
//     client: Client,
//     room: Room,
// ) {
//     if room_member.state_key != client.user_id().unwrap() {
//         return;
//     }

//     tokio::spawn(async move {
//         println!("Autojoining room {}", room.room_id());
//         let mut delay = 2;

//         while let Err(err) = room.join().await {
//             // retry autojoin due to synapse sending invites, before the
//             // invited user can join for more information see
//             // https://github.com/matrix-org/synapse/issues/4345
//             eprintln!("Failed to join room {} ({err:?}), retrying in {delay}s", room.room_id());

//             sleep(Duration::from_secs(delay)).await;
//             delay *= 2;

//             if delay > 3600 {
//                 eprintln!("Can't join room {} ({err:?})", room.room_id());
//                 break;
//             }
//         }
//         if delay <= 3600 {
//             println!("Successfully joined room {}", room.room_id());
//             client.add_event_handler(on_room_message);
//             let banana_desc = call_ai("Explain what a banana is.".to_string()).await.unwrap();
//             println!("AI response: {}", banana_desc);
//             let message = RoomMessageEventContent::text_plain(banana_desc);
//             room.send(message).await.unwrap();
//         }
//     });
// }

// pub async fn login_and_sync() -> anyhow::Result<()> {
    
//     // Note that when encryption is enabled, you should use a persistent store to be
//     // able to restore the session with a working encryption setup.
//     // See the `persist_session` example.
//     let client = Client::builder().homeserver_url(homeserver_url).build().await?;

//     client
//         .matrix_auth()
//         .login_username(&username, &password)
//         .initial_device_display_name("autojoin bot")
//         .await?;

//     println!("logged in as {username}");

//     client.add_event_handler(on_stripped_state_member);

//     client.sync(SyncSettings::default()).await?;

//     Ok(())
// }