from argparse import ArgumentParser

from collections import defaultdict

import boto3

import pdb

from xml.dom.minidom import parseString

import time
from datetime import datetime


def get_hits_info(client):
    hits = client.list_hits(MaxResults=100)['HITs']
    status_to_hits = defaultdict(list)
    for hit in hits:
        status_to_hits[hit['HITStatus']].append(hit['HITId'])
    for hit_status in ['Assignable', 'Unassignable', 'Reviewable', 'Reviewing', 'Disposed']:
        print('[{}]'.format(hit_status))
        for hit_id in status_to_hits[hit_status]:
            print(hit_id)

def expire_hit(client, hit_id):
    past_time = datetime(2015, 1, 1)
    client.update_expiration_for_hit(HITId=hit_id, ExpireAt=past_time)

def expire_all_hits(client):
    hits = client.list_hits(MaxResults=100)['HITs']
    for hit in hits:
        if hit['HITStatus'] in ['Assignable']:
            past_time = datetime(2015, 1, 1)
            client.update_expiration_for_hit(HITId=hit['HITId'], ExpireAt=past_time)

def delete_hit(client, hit_id):
    if hit['HITStatus'] != 'Reviewable':
        expire_hit(client, hit_id)
    client.delete_hit(HITId=hit['HITId'])

def delete_all_hits(client):
    hits = client.list_hits(MaxResults=100)['HITs']
    for hit in hits:
        delete_hit(client, hit['HITId'])

def accept_all_assignments(client):
    hits = client.list_hits(MaxResults=100)['HITs']


def _get_answer_from_assignment(assignment):
    answer_xml = parseString(assignment['Answer'])
    answer = answer_xml.getElementsByTagName('FreeText')[0]
    only_answer = " ".join(t.nodeValue for t in answer.childNodes if t.nodeType == t.TEXT_NODE)
    return only_answer

def review_all_assignments(client, skim_only=False):
    hits = client.list_hits(MaxResults=100)['HITs']
    for hit in hits:
        if hit['HITStatus'] in ['Disposed']:
            continue
        elif hit['HITStatus'] in ['Assignable', 'Unassignable']:
            assignments = client.list_assignments_for_hit(HITId=hit['HITId'])['Assignments']
            print("{} assignments submitted for HITId={} ({} assignments left)".format(len(assignments), hit['HITId'], hit['MaxAssignments'] - len(assignments)))
            if skim_only:
                for assignment in assignments:
                    print(_get_answer_from_assignment(assignment))
        elif hit['HITStatus'] in ['Reviewable']:
            # all HIT assignments are submitted, or HIT expired
            assignments = client.list_assignments_for_hit(HITId=hit['HITId'])['Assignments']
            for assignment in assignments:
                if assignment['AssignmentStatus'] != 'Submitted':
                    continue
                chat_id = _get_answer_from_assignment(assignment)
                if skim_only:
                    print(chat_id)
                else:
                    d = raw_input("https://www.onecommon-tokyo.com/takuma/admin?chat_id={}\nAccept? [Y|N]".format(chat_id))
                    if d == 'Y':
                        client.approve_assignment(AssignmentId=assignment['AssignmentId'])
                    elif d == 'N':
                        requester_feedback = raw_input("reject reason:")
                        client.reject_assignment(AssignmentId=assignment['AssignmentId'], RequesterFeedback=requester_feedback)
                        with open("rejected_chats.txt", "a") as reject_file:
                            reject_file.write(chat_id+"\n")
                    else:
                        continue
        else:
            print("HITId {} is in {} State".format(hit['HITId'], hit['HITStatus']))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sandbox', action='store_true')
    args = parser.parse_args()

    environments = {
        "live": {
            "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
            "preview": "https://www.mturk.com/mturk/preview",
            "manage": "https://requester.mturk.com/mturk/manageHITs",
            "reward": "0.00"
        },
        "sandbox": {
            "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
            "preview": "https://workersandbox.mturk.com/mturk/preview",
            "manage": "https://requestersandbox.mturk.com/mturk/manageHITs",
            "reward": "0.00"
        },
    }
    mturk_environment = environments["live"] if not args.sandbox else environments["sandbox"]

    client = boto3.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'],
    )

    get_hits_info(client)

    #expire_all_hits(client)

    review_all_assignments(client, skim_only=False)


